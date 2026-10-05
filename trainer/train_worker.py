# -*- coding: utf-8 -*-
"""多进程训练器：并行自我对弈 + 训练循环 + 模型热更新

架构：
- 主进程：训练循环（回放池采样 -> 网络训练 -> 保存模型 -> 定期评估）
- 工作进程：自我对弈（GPU 路径：ServerEvaluator 走批量推理服务，不持有模型；
  CPU 路径：本地 DirectEvaluator + 网络副本），样本经队列发回主进程
- GPU 批量推理池：有 GPU 时启动 InferenceServer 进程，攒批统一推理，
  大幅提升 GPU 利用率；无 GPU 时自动回退 CPU，行为与改造前一致
- 模型同步：主进程保存 models/current.pt，推理服务定期检查 mtime 重载；
  CPU 路径 worker 定期重载
"""
import os
import sys
import time
import queue
import multiprocessing as mp
import json
import subprocess
import torch
import torch.nn as nn

# 项目根目录（trainer/ 的上一级），用于定位 tools/update_progress.py
PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRESS_SCRIPT = os.path.join(PROJECT_BASE, 'tools', 'update_progress.py')

from ai.model import XiangqiNet
from ai.mcts import MCTS, BatchMCTS, DirectEvaluator, ServerEvaluator
from trainer.self_play import self_play_game
from trainer.train import ReplayBuffer, train_loop, save_model, load_model
from trainer.evaluate import make_classic_player, evaluate_net
from trainer.inference_server import InferenceServer


def self_play_worker(worker_id: int, sample_queue, model_path: str, stop_event,
                     num_simulations: int = 50, games_before_reload: int = 5,
                     max_moves: int = 200, use_server: bool = False,
                     req_queue=None, resp_queue=None):
    """工作进程：持续自我对弈并发送样本

    GPU 路径（use_server=True）：通过 ServerEvaluator 走 InferenceServer 批量推理，
    worker 不持有模型权重，只做树搜索逻辑（权重同步由服务端按 mtime 完成）。
    CPU 路径（use_server=False）：worker 本地持有网络副本（DirectEvaluator），
    行为与改造前一致（定期重载 current.pt）。
    """
    torch.set_num_threads(2)  # 每进程限制线程数，避免争抢
    if use_server:
        evaluator = ServerEvaluator(req_queue, resp_queue, worker_id)
    else:
        net = XiangqiNet(channels=32, blocks=3)
        # 等待初始模型
        for _ in range(600):
            if os.path.exists(model_path):
                try:
                    load_model(net, model_path)
                    break
                except Exception:
                    time.sleep(1)
            else:
                time.sleep(1)
        evaluator = DirectEvaluator(net)
    mcts = BatchMCTS(evaluator, num_simulations=num_simulations, dirichlet_eps=0.25)
    games = 0
    while not stop_event.is_set():
        try:
            samples = self_play_game(None, mcts, max_moves=max_moves)
            sample_queue.put(samples)
            games += 1
            # 定期重载最新模型（仅 CPU 本地推理路径；服务端路径由 server 同步）
            if not use_server and games % games_before_reload == 0:
                try:
                    load_model(net, model_path)
                except Exception:
                    pass
        except Exception as e:
            # 记录异常避免静默空转（对云端训练同样重要）
            print(f'[worker {worker_id}] 对弈异常 {type(e).__name__}: {e}',
                  file=sys.stderr, flush=True)
            time.sleep(2)


def evaluate_probe(model_path: str, result_queue, num_simulations: int,
                   eval_games: int, channels: int = 32, blocks: int = 3):
    """评估子进程：加载模型 vs 经典基线，结果写入队列（不阻塞训练主循环）"""
    torch.set_num_threads(2)
    try:
        net = XiangqiNet(channels=channels, blocks=blocks)
        load_model(net, model_path)
        baseline = make_classic_player(depth=2, time_limit=0.5)
        stats = evaluate_net(net, baseline, num_games=eval_games,
                             num_simulations=num_simulations, as_red=True)
        result_queue.put(stats)
    except Exception as e:
        result_queue.put({'error': str(e)})


class Trainer:
    """主进程训练器"""

    def __init__(self, model_dir: str, log_dir: str, num_workers: int = 4,
                 num_simulations: int = 50, batch_size: int = 256,
                 lr: float = 1e-3, buffer_capacity: int = 200000,
                 train_every: int = 64, save_every: int = 20,
                 eval_every: int = 50, eval_games: int = 4,
                 progress_every: int = 100,
                 channels: int = 32, blocks: int = 3,
                 resume: bool = True,
                 eval_batch: int = 128, batch_wait: float = 0.02):
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.num_workers = num_workers
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.lr = lr
        self.buffer_capacity = buffer_capacity
        self.train_every = train_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.progress_every = progress_every
        self.channels = channels
        self.blocks = blocks
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, 'current.pt')
        self.best_path = os.path.join(model_dir, 'best.pt')
        self.log_path = os.path.join(log_dir, 'train.log')
        # CUDA 兼容：有 GPU 时训练步跑 GPU，无 GPU 自动回退 CPU（保持原 CPU 行为）
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = XiangqiNet(channels=channels, blocks=blocks).to(self.device)
        # 断点续训：若已有 current.pt 则加载，避免每次从随机初始化
        if resume and os.path.exists(self.model_path):
            try:
                load_model(self.net, self.model_path, device=self.device)
                self._log(f'续训：加载已有模型 {self.model_path} (device={self.device})')
            except Exception:
                self._log('模型加载失败，改用随机初始化')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.sample_queue = mp.Queue(maxsize=200)
        self.stop_event = mp.Event()
        # GPU 批量推理池：有 GPU 时启动 InferenceServer（worker 不再持有模型，
        # 攒批统一推理提升 GPU 利用率）；纯 CPU 时 worker 本地 DirectEvaluator
        self.eval_batch = eval_batch
        self.batch_wait = batch_wait
        self.use_server = torch.cuda.is_available()
        self.req_queue = mp.Queue(maxsize=1024)
        self.resp_queues = {i: mp.Queue() for i in range(num_workers)}
        self.inference_server = None
        if self.use_server:
            self.inference_server = InferenceServer(
                self.model_path, self.req_queue, self.resp_queues,
                channels=channels, blocks=blocks,
                max_batch=eval_batch, batch_wait=batch_wait,
                stop_event=self.stop_event)
            self.inference_server.start()
            self._log(f'已启动 GPU 批量推理服务 (max_batch={eval_batch}, wait={batch_wait}s)')
        self.workers = []
        self.total_games = 0
        self.total_samples = 0
        self.train_steps = 0
        # 异步评估
        self.eval_probe_path = os.path.join(model_dir, 'eval_probe.pt')
        self.eval_queue = mp.Queue()
        self.eval_proc = None
        self._eval_started = False

    def _log(self, msg: str):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def _refresh_progress(self):
        """异步刷新进度页（PNG + HTML），不阻塞训练主循环。

        每 progress_every 步调用一次；上一次子进程未结束时跳过本次，
        避免刷新进程堆积拖慢训练。
        """
        proc = getattr(self, '_progress_proc', None)
        if proc is not None and proc.poll() is None:
            return
        try:
            self._progress_proc = subprocess.Popen(
                [sys.executable, PROGRESS_SCRIPT],
                cwd=PROJECT_BASE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self._log(f'进度页刷新失败（训练继续）: {e}')

    def _notify_baseline_win(self, stats: dict):
        """记录并尝试发送 Windows 通知，避免同一轮评估重复提醒。"""
        alert_path = os.path.join(self.log_dir, 'baseline_win.alert')
        payload = {
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'wins': stats['wins'],
            'draws': stats['draws'],
            'losses': stats['losses'],
            'win_rate': stats['win_rate'],
            'score': stats['score'],
            'model_path': os.path.abspath(self.model_path),
        }
        with open(alert_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 不依赖第三方包；失败时不影响训练，alert 文件仍然保留。
        if os.name == 'nt':
            title = '象棋强化学习'
            message = (
                f'已达到基线胜率：{stats["wins"]}胜 '
                f'{stats["draws"]}和 {stats["losses"]}负'
            )
            script = (
                '[Windows.UI.Notifications.ToastNotificationManager, '
                'Windows.UI.Notifications, ContentType = WindowsRuntime];'
                '$xml = New-Object Windows.Data.Xml.Dom.XmlDocument;'
                '$xml.LoadXml('
                f'\'<toast><visual><binding template="ToastGeneric">'
                f'<text>{title}</text><text>{message}</text>'
                '</binding></visual></toast>\');'
                '$toast = [Windows.UI.Notifications.ToastNotification]::new($xml);'
                '[Windows.UI.Notifications.ToastNotificationManager]::'
                'CreateToastNotifier("Marvis").Show($toast)'
            )
            try:
                subprocess.Popen(
                    ['powershell.exe', '-NoProfile', '-WindowStyle', 'Hidden',
                     '-Command', script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                self._log(f'系统通知发送失败（训练继续）: {exc}')

    def start_workers(self):
        # 模型文件不存在时才保存（续训场景下保留已训练权重）
        if not os.path.exists(self.model_path):
            save_model(self.net, self.model_path)
        for i in range(self.num_workers):
            p = mp.Process(target=self_play_worker, args=(
                i, self.sample_queue, self.model_path, self.stop_event,
                self.num_simulations, 5, 200,
                self.use_server, self.req_queue, self.resp_queues[i]), daemon=True)
            p.start()
            self.workers.append(p)
        self._log(f'启动 {self.num_workers} 个自我对弈进程'
                  f'（推理模式: {"GPU 批量推理服务" if self.use_server else "CPU 本地 DirectEvaluator"}）')

    def stop(self):
        self.stop_event.set()
        for p in self.workers:
            p.terminate()
        self.workers = []
        if self.inference_server is not None:
            # stop_event 会让 server 主循环自然退出；再显式 terminate 兜底
            self.inference_server.terminate()
            self.inference_server = None
        if self.eval_proc is not None:
            self.eval_proc.terminate()
            self.eval_proc = None
        self._eval_started = False

    def _drain_queue(self, timeout: float = 0.5) -> int:
        """从队列收样本，返回新增样本数"""
        added = 0
        while True:
            try:
                samples = self.sample_queue.get(timeout=timeout)
                self.buffer.add(samples)
                added += len(samples)
                self.total_samples += len(samples)
                self.total_games += 1
            except queue.Empty:
                break
        return added

    def _start_evaluate(self):
        """异步启动评估子进程（保存权重快照供其读取，避免与训练竞争）"""
        try:
            save_model(self.net, self.eval_probe_path)
            self._eval_started = True
            self.eval_proc = mp.Process(
                target=evaluate_probe,
                args=(self.eval_probe_path, self.eval_queue,
                      self.num_simulations, self.eval_games),
                daemon=True)
            self.eval_proc.start()
            self._log('评估启动（异步，不阻塞训练）')
        except Exception as e:
            self._eval_started = False
            self._log(f'评估启动失败: {e}')

    def _poll_evaluate(self):
        """轮询异步评估结果"""
        if not self._eval_started:
            return
        try:
            stats = self.eval_queue.get_nowait()
        except queue.Empty:
            return
        self._eval_started = False
        if self.eval_proc is not None:
            self.eval_proc.join(timeout=1)
            self.eval_proc = None
        if 'error' in stats:
            self._log(f'评估失败: {stats["error"]}')
            return
        self._log(f'评估: wins={stats["wins"]} draws={stats["draws"]} '
                  f'losses={stats["losses"]} win_rate={stats["win_rate"]:.2f}')
        if stats['win_rate'] >= 0.55:
            save_model(self.net, self.best_path)
            self._log('胜率达标，已保存为 best.pt')
            self._notify_baseline_win(stats)

    def _ensure_server(self):
        """监控推理服务进程：崩溃后自动用同一队列重启，避免 worker 永久卡死"""
        if not self.use_server:
            return
        if self.inference_server is not None and self.inference_server.is_alive:
            return
        old = self.inference_server
        code = None if old is None else old.exitcode
        # 先确保旧进程完全退出（句柄清理），再重启；立即 spawn 在 Windows
        # 上可能与旧进程残留句柄竞争，导致新进程读不到请求（见 expC4/expC5）
        if old is not None:
            old.join(timeout=5)
        time.sleep(0.5)
        self._log(f'推理服务进程已退出（exitcode={code}），'
                  f'请查看 {os.path.join(self.model_dir, "inference_server_crash.log")}；'
                  f'正在自动重启…')
        self.inference_server = InferenceServer(
            self.model_path, self.req_queue, self.resp_queues,
            channels=self.channels, blocks=self.blocks,
            max_batch=self.eval_batch, batch_wait=self.batch_wait,
            stop_event=self.stop_event)
        self.inference_server.start()
        self._log(f'推理服务已自动重启 (alive={self.inference_server.is_alive})')

    def run(self, max_steps: int = 100000, eval_interval_sec: int = 300):
        """主训练循环（评估异步进行，不阻塞训练）"""
        self.start_workers()
        self._log(f'训练开始: workers={self.num_workers}, sims={self.num_simulations}, '
                  f'batch={self.batch_size}, lr={self.lr}')
        self._refresh_progress()
        last_eval = time.time()
        try:
            while self.train_steps < max_steps and not self.stop_event.is_set():
                # 收样本
                added = self._drain_queue()
                # 样本足够则训练（每步保存/打日志，避免跳号）
                if len(self.buffer) >= self.batch_size and added > 0:
                    # 训练步数与新增样本挂钩：新增样本少时少训，避免抽干
                    # buffer 导致 loss 过拟合到 0（云端样本供给跟不上时，
                    # 固定 train_every=64 会把 buffer 反复抽空）
                    steps = min(self.train_every, max(1, added // self.batch_size))
                    for _ in range(steps):
                        if len(self.buffer) < self.batch_size:
                            break
                        batch = self.buffer.sample(self.batch_size)
                        from trainer.train import train_step
                        stats = train_step(self.net, batch, self.optimizer, device=self.device)
                        self.train_steps += 1
                        if self.train_steps % self.save_every == 0:
                            save_model(self.net, self.model_path)
                            self._log(f'step={self.train_steps} games={self.total_games} '
                                      f'samples={self.total_samples} loss={stats["loss"]:.4f}')
                        # 每 progress_every 步自动刷新进度页（异步，不阻塞训练）
                        if self.train_steps % self.progress_every == 0:
                            self._refresh_progress()
                # 异步评估：到点启动评估进程，训练循环不等待
                if not self._eval_started and time.time() - last_eval > eval_interval_sec:
                    last_eval = time.time()
                    self._start_evaluate()
                self._poll_evaluate()
                # 推理服务崩溃检测与自动重启（含崩溃日志指引）
                self._ensure_server()
                time.sleep(0.05)
        except KeyboardInterrupt:
            self._log('训练中断')
        finally:
            save_model(self.net, self.model_path)
            self._refresh_progress()
            self.stop()
            self._log('训练结束')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='中国象棋 RL 训练器')
    parser.add_argument('--workers', type=int, default=6, help='自我对弈进程数（<=0 自动按 CPU 核数）')
    parser.add_argument('--sims', type=int, default=50, help='MCTS 模拟次数')
    parser.add_argument('--batch', type=int, default=256, help='训练 batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--steps', type=int, default=100000, help='最大训练步数')
    parser.add_argument('--eval-interval', type=int, default=300, help='评估间隔(秒)')
    parser.add_argument('--eval-games', type=int, default=4, help='每次评估对局数')
    parser.add_argument('--progress-every', type=int, default=100, help='每 N 训练步刷新一次进度页')
    parser.add_argument('--eval-batch', type=int, default=128, help='推理服务攒批上限（GPU 批量推理池）')
    parser.add_argument('--batch-wait', type=float, default=0.02, help='推理服务攒批超时窗口（秒）')
    parser.add_argument('--model-dir', type=str, default='models', help='模型目录')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录')
    args = parser.parse_args()

    # 自动检测 CPU 核心数
    if args.workers <= 0:
        args.workers = max(1, os.cpu_count() - 2)
    trainer = Trainer(
        model_dir=args.model_dir, log_dir=args.log_dir,
        num_workers=args.workers, num_simulations=args.sims,
        batch_size=args.batch, lr=args.lr, eval_games=args.eval_games,
        progress_every=args.progress_every,
        eval_batch=args.eval_batch, batch_wait=args.batch_wait,
    )
    trainer.run(max_steps=args.steps, eval_interval_sec=args.eval_interval)


if __name__ == '__main__':
    # 关键：GPU 环境下必须用 spawn 创建子进程（fork 会继承 CUDA 上下文导致
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess）。
    # Windows 默认 spawn；Linux 默认 fork，必须显式切换。
    mp.set_start_method('spawn', force=True)
    main()
