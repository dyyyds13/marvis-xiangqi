# -*- coding: utf-8 -*-
"""InferenceServer 单元测试（本地 CPU 环境验证）

覆盖：
1. 多 worker 并发发请求：攒批聚合、req_id 配对、结果分发正确
2. 攒批聚合统计（batch_requests > 1，总批次数 < 总请求数）
3. mtime 刷新模型（权重 A -> B，推理输出变化证明重载）
4. CPU 环境可运行 + 优雅停止

Windows spawn 安全：worker 模拟函数为模块级，测试主体在 __main__ 保护内。
运行：.venv\Scripts\python.exe tests\test_inference_server.py
"""
import os
import sys
import time

PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_BASE not in sys.path:
    sys.path.insert(0, PROJECT_BASE)

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_server_test_tmp')
MODEL_PATH = os.path.join(TEST_DIR, 'models', 'current.pt')
CHANNELS, BLOCKS = 8, 2  # 小网络加速测试
MOVE_DIM = 8100
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_server_test_log.txt')


def _log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def _worker_sim(worker_id, req_queue, resp_queue, num_rounds, num_boards=2):
    """模拟 self_play worker：连续发送请求并同步校验响应（req_id 配对）"""
    from core.board import XiangqiBoard
    boards = [XiangqiBoard() for _ in range(num_boards)]
    for req_id in range(num_rounds):
        req_queue.put({'worker_id': worker_id, 'req_id': req_id, 'boards': boards})
        resp = resp_queue.get()
        assert resp['req_id'] == req_id, \
            f'worker{worker_id} req_id 配对失败: 期望 {req_id}, 实际 {resp["req_id"]}'
        assert len(resp['ps']) == num_boards, 'ps 数量与请求 boards 不一致'
        assert len(resp['vs']) == num_boards, 'vs 数量与请求 boards 不一致'
        for p in resp['ps']:
            assert len(p) == MOVE_DIM, f'ps 维度错误: {len(p)}'
            assert abs(sum(p) - 1.0) < 0.05, f'策略概率未归一化: {sum(p)}'
        for v in resp['vs']:
            assert -1.0 <= v <= 1.0, f'价值越界: {v}'
    return worker_id


if __name__ == '__main__':
    import shutil
    import multiprocessing as mp
    import torch

    from ai.model import XiangqiNet
    from core.board import XiangqiBoard
    from trainer.train import save_model, load_model
    from trainer.inference_server import InferenceServer

    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    _log('=== InferenceServer 单元测试（CPU 环境） ===')
    _log(f'cuda_available: {torch.cuda.is_available()}')

    # 准备测试目录与初始权重 A
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    net_a = XiangqiNet(channels=CHANNELS, blocks=BLOCKS)
    save_model(net_a, MODEL_PATH)
    _log(f'已保存初始权重 A: {MODEL_PATH}')

    # 创建队列与推理服务
    req_queue = mp.Queue(maxsize=1024)
    num_workers = 3
    resp_queues = {i: mp.Queue() for i in range(num_workers)}
    resp_queues[99] = mp.Queue()  # 额外测试通道（mtime 刷新用）
    stop_event = mp.Event()
    stats_queue = mp.Queue()
    server = InferenceServer(
        MODEL_PATH, req_queue, resp_queues,
        channels=CHANNELS, blocks=BLOCKS,
        max_batch=4, batch_wait=0.05,
        stop_event=stop_event, stats_queue=stats_queue)
    server.start()
    time.sleep(1.0)  # 等 server 加载初始权重
    assert server.is_alive, 'server 未启动'
    _log('server 已启动 (max_batch=4, wait=0.05s)')

    # ---- 测试 1：多 worker 并发，req_id 配对 + 结果分发 ----
    rounds = 5
    procs = []
    for wid in range(num_workers):
        p = mp.Process(target=_worker_sim, args=(wid, req_queue, resp_queues[wid], rounds))
        p.start()
        procs.append(p)
    _log(f'已启动 {num_workers} 个模拟 worker')
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f'worker 模拟进程异常退出: exitcode={p.exitcode}'
    _log(f'[1] 多 worker 并发通过: {num_workers}x{rounds} 轮请求，req_id 配对与分发正确')

    # ---- 测试 2：攒批聚合统计 ----
    batch_reqs = []
    batch_boards = 0
    while not stats_queue.empty():
        st = stats_queue.get()
        batch_reqs.append(st['batch_requests'])
        batch_boards += st['batch_boards']
    total_requests = num_workers * rounds
    total_boards_expected = total_requests * 2
    assert batch_boards == total_boards_expected, \
        f'处理棋盘总数错误: {batch_boards} != {total_boards_expected}'
    assert any(r > 1 for r in batch_reqs), f'未见攒批聚合 (batch_requests={batch_reqs})'
    assert len(batch_reqs) < total_requests, \
        f'未见批量合并: {len(batch_reqs)} 批 vs {total_requests} 请求'
    _log(f'[2] 攒批聚合通过: 共 {len(batch_reqs)} 批 / {total_requests} 请求'
         f'（每批请求数 {sorted(set(batch_reqs))}）')

    # ---- 测试 3：mtime 刷新模型（权重 A -> B，输出变化） ----
    assert server.is_alive, f'server 在测试 2 后已退出 (exitcode={server.exitcode})'
    boards1 = [XiangqiBoard()]
    req_queue.put({'worker_id': 99, 'req_id': 0, 'boards': boards1})
    try:
        resp = resp_queues[99].get(timeout=20)
    except Exception as e:
        _log(f'[3] 获取权重 A 响应失败: {e!r}, alive={server.is_alive}, '
             f'exitcode={server.exitcode}')
        raise
    psA = resp['ps'][0]
    _log(f'[3] 权重 A 推理完成, sum(ps)={sum(psA):.4f}')
    # 生成权重 B 并覆盖 current.pt（mtime 变化）。
    # 注意：不能用"另一个随机初始化网络"作为 B——softmax 输出接近均匀分布，
    # 与权重 A 的输出差异极小（~1e-5），无法验证刷新。改为对最后一层 bias 注入
    # 显著信号：softmax 后目标动作概率接近 1，与均匀分布差异巨大。
    net_b = XiangqiNet(channels=CHANNELS, blocks=BLOCKS)
    with torch.no_grad():
        net_b.policy_fc2.bias.fill_(0.0)
        net_b.policy_fc2.bias[321] = 10.0  # 指定动作 logit 拉高
        net_b.value_fc2.bias.fill_(0.5)    # 价值输出也显著偏移
    save_model(net_b, MODEL_PATH)
    _log('[3] 已覆盖权重 B（bias 注入信号），等待 server 重载...')
    time.sleep(1.5)  # 等 server 检测 mtime 并重载
    req_queue.put({'worker_id': 99, 'req_id': 1, 'boards': boards1})
    try:
        resp = resp_queues[99].get(timeout=20)
    except Exception as e:
        _log(f'[3] 获取权重 B 响应失败: {e!r}, alive={server.is_alive}, '
             f'exitcode={server.exitcode}')
        raise
    psB = resp['ps'][0]
    max_diff = max(abs(a - b) for a, b in zip(psA, psB))
    assert max_diff > 0.01, f'权重未刷新（输出几乎不变 max_diff={max_diff:.6f}）'
    _log(f'[3] mtime 刷新通过: 重载后推理输出变化 max_diff={max_diff:.4f}')

    # ---- 测试 4：优雅停止 ----
    stop_event.set()
    server.join(timeout=8)
    assert not server.is_alive, 'server 未随 stop_event 退出'
    _log('[4] 优雅停止通过')

    # 清理测试资源
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    _log('=== 全部测试通过 ===')
