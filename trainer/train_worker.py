# -*- coding: utf-8 -*-
"""多进程训练器：并行自我对弈 + 训练循环 + 模型热更新

架构：
- 主进程：训练循环（回放池采样 -> 网络训练 -> 保存模型 -> 定期评估）
- 工作进程：自我对弈（每进程独立 MCTS + 网络副本），样本经队列发回主进程
- 模型同步：主进程保存 models/current.pt，工作进程定期检查 mtime 重载
"""
import os
import time
import queue
import multiprocessing as mp
import torch
import torch.nn as nn

from ai.model import XiangqiNet
from ai.mcts import MCTS
from trainer.self_play import self_play_game
from trainer.train import ReplayBuffer, train_loop, save_model, load_model
from trainer.evaluate import make_classic_player, evaluate_net


def self_play_worker(worker_id: int, sample_queue, model_path: str, stop_event,
                     num_simulations: int = 50, games_before_reload: int = 5,
                     max_moves: int = 200):
    """工作进程：持续自我对弈并发送样本"""
    torch.set_num_threads(2)  # 每进程限制线程数，避免争抢
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
    mcts = MCTS(net, num_simulations=num_simulations, dirichlet_eps=0.25)
    games = 0
    while not stop_event.is_set():
        try:
            samples = self_play_game(net, mcts, max_moves=max_moves)
            sample_queue.put(samples)
            games += 1
            # 定期重载最新模型
            if games % games_before_reload == 0:
                try:
                    load_model(net, model_path)
                except Exception:
                    pass
        except Exception as e:
            time.sleep(2)


class Trainer:
    """主进程训练器"""

    def __init__(self, model_dir: str, log_dir: str, num_workers: int = 4,
                 num_simulations: int = 50, batch_size: int = 256,
                 lr: float = 1e-3, buffer_capacity: int = 200000,
                 train_every: int = 64, save_every: int = 20,
                 eval_every: int = 50, eval_games: int = 10,
                 channels: int = 32, blocks: int = 3):
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
        self.channels = channels
        self.blocks = blocks
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, 'current.pt')
        self.best_path = os.path.join(model_dir, 'best.pt')
        self.log_path = os.path.join(log_dir, 'train.log')
        self.net = XiangqiNet(channels=channels, blocks=blocks)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.sample_queue = mp.Queue(maxsize=200)
        self.stop_event = mp.Event()
        self.workers = []
        self.total_games = 0
        self.total_samples = 0
        self.train_steps = 0

    def _log(self, msg: str):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def start_workers(self):
        save_model(self.net, self.model_path)
        for i in range(self.num_workers):
            p = mp.Process(target=self_play_worker, args=(
                i, self.sample_queue, self.model_path, self.stop_event,
                self.num_simulations), daemon=True)
            p.start()
            self.workers.append(p)
        self._log(f'启动 {self.num_workers} 个自我对弈进程')

    def stop(self):
        self.stop_event.set()
        for p in self.workers:
            p.terminate()
        self.workers = []

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

    def _evaluate(self) -> dict:
        """评估当前模型 vs 经典基线"""
        baseline = make_classic_player(depth=2, time_limit=0.5)
        stats = evaluate_net(self.net, baseline, num_games=self.eval_games,
                             num_simulations=self.num_simulations, as_red=True)
        return stats

    def run(self, max_steps: int = 100000, eval_interval_sec: int = 300):
        """主训练循环"""
        self.start_workers()
        self._log(f'训练开始: workers={self.num_workers}, sims={self.num_simulations}, '
                  f'batch={self.batch_size}, lr={self.lr}')
        last_eval = time.time()
        try:
            while self.train_steps < max_steps:
                # 收样本
                added = self._drain_queue()
                # 样本足够则训练
                if len(self.buffer) >= self.batch_size and added > 0:
                    for _ in range(self.train_every):
                        if len(self.buffer) < self.batch_size:
                            break
                        batch = self.buffer.sample(self.batch_size)
                        from trainer.train import train_step
                        stats = train_step(self.net, batch, self.optimizer)
                        self.train_steps += 1
                    if self.train_steps % self.save_every == 0:
                        save_model(self.net, self.model_path)
                        self._log(f'step={self.train_steps} games={self.total_games} '
                                  f'samples={self.total_samples} loss={stats["loss"]:.4f}')
                # 定期评估
                if time.time() - last_eval > eval_interval_sec:
                    last_eval = time.time()
                    stats = self._evaluate()
                    self._log(f'评估: wins={stats["wins"]} draws={stats["draws"]} '
                              f'losses={stats["losses"]} win_rate={stats["win_rate"]:.2f}')
                    if stats['win_rate'] >= 0.55:
                        save_model(self.net, self.best_path)
                        self._log('胜率达标，已保存为 best.pt')
                time.sleep(0.1)
        except KeyboardInterrupt:
            self._log('训练中断')
        finally:
            save_model(self.net, self.model_path)
            self.stop()
            self._log('训练结束')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='中国象棋 RL 训练器')
    parser.add_argument('--workers', type=int, default=4, help='自我对弈进程数')
    parser.add_argument('--sims', type=int, default=50, help='MCTS 模拟次数')
    parser.add_argument('--batch', type=int, default=256, help='训练 batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--steps', type=int, default=100000, help='最大训练步数')
    parser.add_argument('--eval-interval', type=int, default=300, help='评估间隔(秒)')
    parser.add_argument('--model-dir', type=str, default='models', help='模型目录')
    parser.add_argument('--log-dir', type=str, default='logs', help='日志目录')
    args = parser.parse_args()

    # 自动检测 CPU 核心数
    if args.workers <= 0:
        args.workers = max(1, os.cpu_count() - 1)
    trainer = Trainer(
        model_dir=args.model_dir, log_dir=args.log_dir,
        num_workers=args.workers, num_simulations=args.sims,
        batch_size=args.batch, lr=args.lr,
    )
    trainer.run(max_steps=args.steps, eval_interval_sec=args.eval_interval)


if __name__ == '__main__':
    main()
