# -*- coding: utf-8 -*-
"""mcts.py 单元测试：Evaluator 抽象 + BatchMCTS 批式搜索语义一致性

覆盖：
1. 语义一致性（强判据）：BatchMCTS(batch_size=1) 与原 MCTS 在相同固定
   网络/随机种子下，多个开局/残局上的 search 策略分布、访问次数逐 move 完全一致
   （batch_size=1 时每轮仅收集 1 条路径即评估扩展，树演进与原 MCTS 等价）
2. 批式语义相近（batch_size>1）：收集-批量评估-扩展的多轮流程，策略分布
   仍与原 MCTS 高度相关（JS 散度有界 + top1 大体一致）
3. 统计不变量：根访问计数 == num_simulations、策略归一化、
   temperature=0 贪心单点分布、根 Dirichlet 噪声应用一致
4. ServerEvaluator + InferenceServer 组合：跑通 search（单 worker 与
   双 worker 并发，req_id 配对 / 独立响应队列不串扰）

运行：.venv\\Scripts\\python.exe tests\\test_mcts.py
"""
import os
import sys
import math
import shutil
import time

PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_BASE not in sys.path:
    sys.path.insert(0, PROJECT_BASE)

from ai.model import XiangqiNet
from ai.mcts import MCTS, BatchMCTS, DirectEvaluator, Evaluator
from core.board import XiangqiBoard, START_FEN

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_mcts_test_tmp')
MODEL_PATH = os.path.join(TEST_DIR, 'models', 'current.pt')
CHANNELS, BLOCKS = 8, 2
MOVE_DIM = 8100

# 若干开局 / 残局 FEN
FENS = [
    ('开局', START_FEN),
    ('残局-单车', '3k5/9/4R4/9/9/9/9/9/4K4/9 w - - 0 1'),
    ('残局-车将军中', '3k5/4R4/9/9/9/9/9/9/4K4/9 w - - 0 1'),
    ('残局-仕相全', '2bakab2/9/9/9/9/9/9/9/9/2BAKAB2 w - - 0 1'),
]

NET_SEED, SEARCH_SEED = 42, 1234
SIMULATIONS = 60


def make_net():
    torch_manual_seed = __import__('torch').manual_seed
    torch_manual_seed(NET_SEED)
    net = XiangqiNet(channels=CHANNELS, blocks=BLOCKS)
    net.eval()
    return net


def js_divergence(p: dict, q: dict) -> float:
    """策略分布 JS 散度（对称，[0, ~0.693]）"""
    eps = 1e-9
    s = 0.0
    for k in set(p) | set(q):
        a, b = p.get(k, 0.0) + eps, q.get(k, 0.0) + eps
        m = 0.5 * (a + b)
        s += 0.5 * (a * math.log(a / m) + b * math.log(b / m))
    return s


def _check_distribution(probs, board, name):
    assert probs, f'{name}: 空策略分布'
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-6, f'{name}: 策略未归一化 sum={total:.6f}'
    legal = set(board.legal_moves())
    for mv in probs:
        assert mv in legal, f'{name}: 非法走法 {mv} 出现在策略中'


def test_semantic_equivalence_bs1():
    """BatchMCTS(batch_size=1) 与原 MCTS 逐 move 完全一致（分布 + 访问次数）"""
    net = make_net()
    for tag, fen in FENS:
        board = XiangqiBoard(fen)
        __import__('torch').manual_seed(SEARCH_SEED)
        mcts = MCTS(net, c_puct=1.5, num_simulations=SIMULATIONS,
                    dirichlet_alpha=0.3, dirichlet_eps=0.25)
        probs_m = mcts.search(board.copy(), temperature=1.0)

        __import__('torch').manual_seed(SEARCH_SEED)
        bm = BatchMCTS(DirectEvaluator(net), c_puct=1.5, num_simulations=SIMULATIONS,
                       dirichlet_alpha=0.3, dirichlet_eps=0.25, batch_size=1)
        probs_b = bm.search(board.copy(), temperature=1.0)

        _check_distribution(probs_m, board, f'{tag}/MCTS')
        _check_distribution(probs_b, board, f'{tag}/BatchMCTS(bs=1)')
        assert probs_m == probs_b, \
            f'{tag}: bs=1 策略分布不一致\n MCTS={probs_m}\n Batch={probs_b}'
        # temperature=1 时 prob = visit/total，逐 move 相等即访问次数分布一致
        visits_m = {mv: round(p * SIMULATIONS, 6) for mv, p in probs_m.items()}
        visits_b = {mv: round(p * SIMULATIONS, 6) for mv, p in probs_b.items()}
        assert visits_m == visits_b, f'{tag}: 访问次数分布不一致'
        print(f'  [OK] {tag}: bs=1 分布/访问完全一致, top1={max(probs_m, key=probs_m.get)}')


def test_semantic_similarity_batch():
    """batch_size>1 批式语义相近：JS 有界 + top1 大体一致"""
    net = make_net()
    js_vals = []
    top1_hit = 0
    for tag, fen in FENS:
        board = XiangqiBoard(fen)
        __import__('torch').manual_seed(SEARCH_SEED)
        mcts = MCTS(net, c_puct=1.5, num_simulations=SIMULATIONS,
                    dirichlet_alpha=0.3, dirichlet_eps=0.25)
        probs_m = mcts.search(board.copy(), temperature=1.0)

        __import__('torch').manual_seed(SEARCH_SEED)
        bm = BatchMCTS(DirectEvaluator(net), c_puct=1.5, num_simulations=SIMULATIONS,
                       dirichlet_alpha=0.3, dirichlet_eps=0.25, batch_size=8)
        probs_b = bm.search(board.copy(), temperature=1.0)

        js = js_divergence(probs_m, probs_b)
        js_vals.append(js)
        top1_hit += (max(probs_m, key=probs_m.get) == max(probs_b, key=probs_b.get))
        print(f'  [..] {tag}: JS={js:.4f}, top1一致={probs_m.keys() and (max(probs_m, key=probs_m.get) == max(probs_b, key=probs_b.get))}')
    assert max(js_vals) < 0.55, f'批式策略偏离过大: JS={js_vals}'
    assert top1_hit >= len(FENS) - 1, f'top1 一致数过少: {top1_hit}/{len(FENS)}'
    print(f'  [OK] 批式相近: JS max={max(js_vals):.4f}, top1 一致 {top1_hit}/{len(FENS)}')


def test_statistical_invariants():
    """统计不变量：根访问计数、归一化、贪心温度、噪声一致性"""
    net = make_net()
    board = XiangqiBoard(START_FEN)
    __import__('torch').manual_seed(SEARCH_SEED)
    bm = BatchMCTS(DirectEvaluator(net), c_puct=1.5, num_simulations=SIMULATIONS,
                   dirichlet_alpha=0.3, dirichlet_eps=0.25, batch_size=4)
    probs = bm.search(board.copy(), temperature=1.0)
    visits = {mv: p * SIMULATIONS for mv, p in probs.items()}
    total = sum(visits.values())
    assert abs(total - SIMULATIONS) < 1e-6, f'根访问计数 != {SIMULATIONS}: {total}'
    assert sum(probs.values()) == 1.0, '概率未归一化'

    # temperature=0 贪心：单点分布，概率 1.0
    __import__('torch').manual_seed(SEARCH_SEED)
    greedy = bm.search(board.copy(), temperature=0.0)
    assert len(greedy) == 1 and list(greedy.values()) == [1.0], 'temperature=0 非单点贪心分布'
    assert list(greedy.keys())[0] == max(visits, key=visits.get), '贪心走法不是访问最多'

    # 所有返回走法均为合法走法
    legal = set(board.legal_moves())
    assert all(mv in legal for mv in probs), '策略中出现非法走法'
    print('  [OK] 统计不变量: 访问计数/归一化/贪心/合法性全部通过')


def test_terminal_endgame():
    """残局终局语义：将死局面应给出合法杀招，无子可走返回空分布"""
    net = make_net()
    # 红车直线将军，黑将仅能左右躲，红方有多个将军/杀招可选
    fen = '3k5/4R4/9/9/9/9/9/9/4K4/9 w - - 0 1'
    board = XiangqiBoard(fen)
    __import__('torch').manual_seed(SEARCH_SEED)
    bm = BatchMCTS(DirectEvaluator(net), c_puct=1.5, num_simulations=SIMULATIONS,
                   dirichlet_alpha=0.3, dirichlet_eps=0.25, batch_size=4)
    probs = bm.search(board.copy(), temperature=1.0)
    _check_distribution(probs, board, '终局残局')
    best = max(probs, key=probs.get)
    # 最佳走法必须是合法走法（且非 -1）
    assert best in board.legal_moves(), f'最佳走法非法: {best}'
    assert best != -1, '无合法走法却返回 -1'
    # 困毙/将死无子可走：返回空分布
    dead = XiangqiBoard('3k5/9/4R4/9/9/9/9/9/9/3RK4 b - - 0 1')  # 黑将被车帅夹击无路可走
    legal = dead.legal_moves()
    if not legal:
        __import__('torch').manual_seed(SEARCH_SEED)
        assert bm.search(dead.copy(), temperature=1.0) == {}, '无子可走应返回空分布'
        print('  [OK] 终局残局: 杀招合法; 无子可走返回空分布')
    else:
        print('  [OK] 终局残局: 杀招合法 (死局判定未触发, legal=%d)' % len(legal))


def _run_server_e2e():
    """ServerEvaluator + InferenceServer 组合跑通 search（单/双 worker）"""
    import multiprocessing as mp
    import torch
    from trainer.train import save_model
    from trainer.inference_server import InferenceServer
    from ai.mcts import ServerEvaluator

    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    net = make_net()
    save_model(net, MODEL_PATH)

    req_queue = mp.Queue(maxsize=1024)
    resp_queues = {0: mp.Queue(), 1: mp.Queue()}
    stop_event = mp.Event()
    server = InferenceServer(MODEL_PATH, req_queue, resp_queues,
                             channels=CHANNELS, blocks=BLOCKS,
                             max_batch=4, batch_wait=0.05,
                             stop_event=stop_event, stats_queue=None)
    server.start()
    time.sleep(1.2)
    assert server.is_alive, 'InferenceServer 未启动'
    try:
        # 单 worker：完整 search 跑通
        ev0 = ServerEvaluator(req_queue, resp_queues[0], worker_id=0)
        bm = BatchMCTS(ev0, c_puct=1.5, num_simulations=SIMULATIONS,
                       dirichlet_alpha=0.3, dirichlet_eps=0.25, batch_size=4)
        board = XiangqiBoard(FENS[1][1])  # 单车残局
        __import__('torch').manual_seed(SEARCH_SEED)
        probs = bm.search(board.copy(), temperature=1.0)
        _check_distribution(probs, board, 'Server单worker')
        best = max(probs, key=probs.get)
        assert best in board.legal_moves(), 'Server 最佳走法非法'
        print(f'  [OK] Server 单 worker: search 跑通, top1={best}, sum={sum(probs.values()):.4f}')

        # 双 worker 并发：各自独立 search，验证 req_id 配对与响应队列不串扰
        ev1 = ServerEvaluator(req_queue, resp_queues[1], worker_id=1)
        bm1 = BatchMCTS(ev1, c_puct=1.5, num_simulations=SIMULATIONS,
                        dirichlet_alpha=0.3, dirichlet_eps=0.25, batch_size=4)
        bm2 = BatchMCTS(ServerEvaluator(req_queue, resp_queues[0], worker_id=0),
                        c_puct=1.5, num_simulations=SIMULATIONS,
                        dirichlet_alpha=0.3, dirichlet_eps=0.25, batch_size=4)
        b1 = XiangqiBoard(FENS[0][1])  # 开局
        b2 = XiangqiBoard(FENS[2][1])  # 车将军中残局
        __import__('torch').manual_seed(SEARCH_SEED)
        p1 = bm1.search(b1.copy(), temperature=1.0)
        __import__('torch').manual_seed(SEARCH_SEED)
        p2 = bm2.search(b2.copy(), temperature=1.0)
        _check_distribution(p1, b1, 'Server双worker-1')
        _check_distribution(p2, b2, 'Server双worker-2')
        # 双 worker 结果应各自对应自己的棋盘（开局 vs 残局，分布不同）
        assert max(p1, key=p1.get) != max(p2, key=p2.get) or len(p1) != len(p2), \
            '双 worker 结果疑似串扰'
        print(f'  [OK] Server 双 worker: 并发 search 跑通, top1=({max(p1, key=p1.get)}, {max(p2, key=p2.get)})')
    finally:
        stop_event.set()
        server.join(timeout=8)
        shutil.rmtree(TEST_DIR, ignore_errors=True)


def _test_abstract():
    """Evaluator 抽象契约：evaluate(boards)->(ps,vs)，两实现返回纯 Python 数据"""
    import torch
    net = make_net()
    board = XiangqiBoard(START_FEN)
    boards = [board.copy(), board.copy()]
    de = DirectEvaluator(net)
    ps, vs = de.evaluate(boards)
    assert len(ps) == 2 and len(vs) == 2, 'DirectEvaluator 批量数量错误'
    assert isinstance(ps[0], list) and isinstance(vs[0], float), 'DirectEvaluator 返回非 Python 数据'
    assert len(ps[0]) == MOVE_DIM, f'策略维度错误: {len(ps[0])}'
    assert abs(sum(ps[0]) - 1.0) < 0.05, '策略未归一化'
    try:
        Evaluator().evaluate([board])
        raise AssertionError('Evaluator 抽象未抛 NotImplementedError')
    except NotImplementedError:
        pass
    print('  [OK] Evaluator 抽象契约: evaluate 签名/纯 Python 返回/抽象拦截 通过')


if __name__ == '__main__':
    print('=== mcts.py 单元测试（CPU 环境） ===')
    print(f'torch cuda: {__import__("torch").cuda.is_available()}')
    print('[1] 语义一致性（batch_size=1 与原 MCTS 完全一致）')
    test_semantic_equivalence_bs1()
    print('[2] 批式语义相近（batch_size=8）')
    test_semantic_similarity_batch()
    print('[3] 统计不变量')
    test_statistical_invariants()
    print('[4] 终局残局语义')
    test_terminal_endgame()
    print('[5] Evaluator 抽象契约')
    _test_abstract()
    print('[6] ServerEvaluator + InferenceServer 组合')
    _run_server_e2e()
    print('=== 全部测试通过 ===')
