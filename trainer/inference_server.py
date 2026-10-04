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
"""
import os
import time
import queue
import multiprocessing as mp

import torch

from ai.model import XiangqiNet, board_to_tensor
from trainer.train import load_model


class InferenceServer(mp.Process):
    """批量推理服务进程"""

    def __init__(self, model_path: str, req_queue, resp_queues,
                 channels: int = 32, blocks: int = 3,
                 max_batch: int = 128, batch_wait: float = 0.02,
                 stop_event=None, stats_queue=None):
        super().__init__(daemon=True)
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _model_mtime(self):
        try:
            return os.path.getmtime(self.model_path)
        except OSError:
            return None

    def run(self):
        torch.set_num_threads(4)
        net = XiangqiNet(channels=self.channels, blocks=self.blocks).to(self.device)
        # 等待初始模型文件出现（训练主进程 start_workers 会先保存初始权重）
        while not self.stop_event.is_set():
            if os.path.exists(self.model_path):
                try:
                    load_model(net, self.model_path, device=self.device)
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                time.sleep(0.2)
        net.eval()
        last_mtime = self._model_mtime()
        pending = []  # [(worker_id, req_id, boards), ...]
        while not self.stop_event.is_set():
            # ---- 攒批收集：直到 max_batch 或 batch_wait 超时 ----
            deadline = time.monotonic() + self.batch_wait
            while len(pending) < self.max_batch:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                try:
                    item = self.req_queue.get(timeout=remain)
                except queue.Empty:
                    break
                pending.append(item)
                if len(pending) >= self.max_batch:
                    break
            # ---- 模型热更新：mtime 变化则重新加载（失败保留旧权重，下轮重试）----
            mt = self._model_mtime()
            if mt is not None and mt != last_mtime:
                try:
                    load_model(net, self.model_path, device=self.device)
                    net.eval()
                    last_mtime = mt
                except Exception:
                    pass  # 训练进程可能正在写文件，保留旧权重下轮再试
            # ---- 统一推理并分发 ----
            if pending:
                self._process_batch(net, pending)
                pending = []
        # 退出前清空已收集的剩余请求（stop 场景下 worker 即将被终止）
        if pending:
            try:
                self._process_batch(net, pending)
            except Exception:
                pass

    def _process_batch(self, net, pending):
        """合并本批所有请求的棋盘，一次推理后按 worker 分发"""
        total_boards = sum(len(item[2]) for item in pending)
        tensors = torch.stack(
            [board_to_tensor(b) for item in pending for b in item[2]]
        ).to(self.device)
        with torch.no_grad():
            p, v = net(tensors)
            p = torch.softmax(p, dim=1).cpu()  # CUDA 张量禁止进 mp.Queue，转 CPU
            v = v.squeeze(1).cpu()
        idx = 0
        for worker_id, req_id, boards in pending:
            n = len(boards)
            resp = {
                'req_id': req_id,
                'ps': p[idx:idx + n].tolist(),   # Python 数值（float list）
                'vs': v[idx:idx + n].tolist(),
            }
            self.resp_queues[worker_id].put(resp)
            idx += n
        if self.stats_queue is not None:
            self.stats_queue.put({'batch_boards': total_boards, 'batch_requests': len(pending)})
