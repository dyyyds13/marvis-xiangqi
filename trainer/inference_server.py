# -*- coding: utf-8 -*-
"""GPU 批量推理服务进程（Leela Chess Zero 式推理池）

架构：
- 独立 mp.Process，持有 GPU 模型副本（XiangqiNet, device=cuda）
- 通过请求队列接收各自我对弈 worker 的评估请求，攒批后统一 predict_batch，
  结果分发回各 worker 的独立响应队列
- 请求格式  : {"worker_id": int, "req_id": int, "boards": [XiangqiBoard, ...]}
- 响应格式  : {"req_id": int, "ps": [[float x 8100], ...], "vs": [float, ...]}
- 攒批策略  : 收集请求直到 max_batch 或 batch_wait 超时窗口到达即推理
- 模型同步  : 定期检查 models/current.pt 的 mtime，变化则重新 load_model

安全约束：
- CUDA 张量绝不进 mp.Queue：跨进程只传 XiangqiBoard 对象与 Python 数值
- 推理在 server 进程内完成，结果 .cpu() 转 Python list 后分发

实现说明：
- 服务以普通 mp.Process(target=...) 方式启动（模块级函数 _server_run），
  不用 mp.Process 子类：Windows spawn 下子类实例 pickle 会因 _authkey
  （AuthenticationString 禁止 pickle）崩溃。InferenceServer 类只是薄封装，
  对外提供 start()/join()/terminate()/is_alive/exitcode 接口。
"""
import os
import sys
import time
import queue
import multiprocessing as mp

import torch

from ai.model import XiangqiNet, board_to_tensor
from trainer.train import load_model


def _server_run(model_path, req_queue, resp_queues,
                channels, blocks, max_batch, batch_wait,
                stop_event, stats_queue):
    """推理服务进程主体（模块级函数，Windows spawn 可安全 pickle 参数）"""
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debug = os.environ.get('MARVIS_SERVER_DEBUG', '') == '1'

    def _dbg(msg):
        if debug:
            print(f'[server-debug] {msg}', file=sys.stderr, flush=True)

    def _model_mtime():
        try:
            return os.path.getmtime(model_path)
        except OSError:
            return None

    try:
        net = XiangqiNet(channels=channels, blocks=blocks).to(device)
        # 等待初始模型文件出现（训练主进程 start_workers 会先保存初始权重）
        while not stop_event.is_set():
            if os.path.exists(model_path):
                try:
                    load_model(net, model_path, device=device)
                    break
                except Exception:
                    _dbg('init load fail, retry')
                    time.sleep(0.2)
            else:
                time.sleep(0.2)
        net.eval()
        _dbg(f'[pid={os.getpid()}] entered main loop')
        # 启动延迟：terminate 旧进程后立即重启的新进程若过早读共享 mp.Queue，
        # Windows spawn 下可能读不到数据（句柄清理竞态，见 expCq）。进入主
        # 循环前先让出 1s，等待旧进程句柄完全释放（本地 expCq 系列验证有效）。
        time.sleep(1.0)
        last_mtime = _model_mtime()
        pending = []  # [{"worker_id": int, "req_id": int, "boards": [XiangqiBoard,...]}, ...]
        while not stop_event.is_set():
            # ---- 攒批收集：直到 max_batch 或 batch_wait 超时 ----
            deadline = time.monotonic() + batch_wait
            while len(pending) < max_batch:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                try:
                    item = req_queue.get(timeout=remain)
                except queue.Empty:
                    break
                pending.append(item)
                _dbg(f'got req worker={item["worker_id"]} pending={len(pending)}')
                if len(pending) >= max_batch:
                    break
            # ---- 模型热更新：mtime 变化则重新加载（失败保留旧权重，下轮重试）----
            mt = _model_mtime()
            if mt is not None and mt != last_mtime:
                try:
                    load_model(net, model_path, device=device)
                    net.eval()
                    last_mtime = mt
                    _dbg('hot reload done')
                except Exception:
                    pass  # 训练进程可能正在写文件，保留旧权重下轮再试
            # ---- 统一推理并分发 ----
            if pending:
                _dbg(f'process batch {len(pending)}')
                _process_batch(net, pending, resp_queues, stats_queue, device)
                pending = []
                _dbg('batch responded')
            # 让步：避免忙循环空转烧 CPU；同时规避 Windows spawn 下同一
            # mp.Queue 被 terminate 后重启进程时读端句柄竞争的竞态（见
            # expD/expF/expCq）。本地 Windows CPU 场景用 0.05 验证稳定；
            # 云端 GPU 推理仅数毫秒，0.05 的固定 sleep 会把单请求延迟拉高
            # 到 70ms+，8 worker 串行请求下吞吐骤降、样本枯竭、loss 过拟合
            # 到 0。云端 Linux 无 Windows 句柄竞态，0.002 足够让步且不拖
            # 吞吐；Windows 场景可经环境变量调回（见下）。
            time.sleep(float(os.environ.get('MARVIS_SERVER_SLEEP', '0.002')))
        # 退出前清空已收集的剩余请求（stop 场景下 worker 即将被终止）
        if pending:
            try:
                _process_batch(net, pending, resp_queues, stats_queue, device)
            except Exception:
                pass
    except BaseException:
        # 服务进程异常崩溃必须可见：stderr + 崩溃日志文件
        import traceback
        tb = traceback.format_exc()
        sys.stderr.write(f'[InferenceServer] 崩溃:\n{tb}\n')
        sys.stderr.flush()
        crash_log = os.path.join(
            os.path.dirname(model_path) or '.', 'inference_server_crash.log')
        try:
            with open(crash_log, 'a', encoding='utf-8') as f:
                f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] 崩溃:\n{tb}\n')
        except Exception:
            pass
        raise


def _process_batch(net, pending, resp_queues, stats_queue, device):
    """合并本批所有请求的棋盘，一次推理后按 worker 分发"""
    total_boards = sum(len(item['boards']) for item in pending)
    tensors = torch.stack(
        [board_to_tensor(b) for item in pending for b in item['boards']]
    ).to(device)
    with torch.no_grad():
        p, v = net(tensors)
        p = torch.softmax(p, dim=1).cpu()  # CUDA 张量禁止进 mp.Queue，转 CPU
        v = v.squeeze(1).cpu()
    idx = 0
    for item in pending:
        worker_id, req_id = item['worker_id'], item['req_id']
        n = len(item['boards'])
        resp = {
            'req_id': req_id,
            'ps': p[idx:idx + n].tolist(),   # Python 数值（float list）
            'vs': v[idx:idx + n].tolist(),
        }
        resp_queues[worker_id].put(resp)
        idx += n
    if stats_queue is not None:
        stats_queue.put({'batch_boards': total_boards, 'batch_requests': len(pending)})


class InferenceServer:
    """GPU 批量推理服务进程（薄封装：内部为标准 mp.Process(target=_server_run)）"""

    def __init__(self, model_path: str, req_queue, resp_queues,
                 channels: int = 32, blocks: int = 3,
                 max_batch: int = 128, batch_wait: float = 0.02,
                 stop_event=None, stats_queue=None):
        self.model_path = model_path
        self.req_queue = req_queue
        # resp_queues: dict {worker_id: mp.Queue}，每个 worker 独立响应队列，
        # 避免多 worker 共享队列时响应被交叉消费导致死锁
        self.resp_queues = resp_queues
        self.channels = channels
        self.blocks = blocks
        self.max_batch = max_batch
        self.batch_wait = batch_wait
        self.stop_event = stop_event if stop_event is not None else mp.Event()
        # 可选：测试用，每批处理后回传批统计（生产不传）
        self.stats_queue = stats_queue
        self._proc = None

    def start(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = mp.Process(
            target=_server_run,
            args=(self.model_path, self.req_queue, self.resp_queues,
                  self.channels, self.blocks, self.max_batch, self.batch_wait,
                  self.stop_event, self.stats_queue),
            daemon=True)
        self._proc.start()

    def join(self, timeout=None):
        if self._proc is not None:
            self._proc.join(timeout)

    def terminate(self):
        if self._proc is not None:
            self._proc.terminate()

    @property
    def is_alive(self):
        return self._proc is not None and self._proc.is_alive()

    @property
    def exitcode(self):
        return None if self._proc is None else self._proc.exitcode
