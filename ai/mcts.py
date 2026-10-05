# -*- coding: utf-8 -*-
"""AlphaZero 式 MCTS：UBC 公式 + 虚拟损失 + 转置表"""
import math
import random
import torch
import torch.nn.functional as F
from core.board import XiangqiBoard, RED, BLACK
from core.rules import would_be_illegal_cycle
from ai.model import XiangqiNet, legal_move_mask


class MCTSNode:
    __slots__ = ('prior', 'visit_count', 'total_value', 'children', 'legal_moves', 'parent', 'move')

    def __init__(self, prior: float, parent=None, move=None):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children = {}   # move -> MCTSNode
        self.legal_moves = None
        self.parent = parent
        self.move = move

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_expanded(self) -> bool:
        return self.legal_moves is not None


class MCTS:
    def __init__(self, net: XiangqiNet, c_puct: float = 1.5, num_simulations: int = 200,
                 virtual_loss: int = 1, dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25):
        self.net = net
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.virtual_loss = virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.transposition = {}  # hash -> node

    def _uct(self, node: MCTSNode, child: MCTSNode) -> float:
        """UBC 公式（从当前方视角）"""
        q = -child.q_value  # 子节点价值是对方视角，取反
        u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        return q + u

    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择：沿 UCT 最大的子节点下降，直到叶节点"""
        while node.is_expanded() and node.children:
            best_move = None
            best_val = -float('inf')
            for mv, child in node.children.items():
                val = self._uct(node, child)
                if val > best_val:
                    best_val = val
                    best_move = mv
            node = node.children[best_move]
        return node

    def _expand(self, board: XiangqiBoard, node: MCTSNode):
        """展开叶节点：获取合法走法 + 网络先验"""
        moves = board.legal_moves()
        node.legal_moves = moves
        if not moves:
            return
        # 棋规约束：剔除会导致自己长将/长捉/长杀违规的走法
        ok_moves = []
        for mv in moves:
            if not would_be_illegal_cycle(board, mv):
                ok_moves.append(mv)
        if ok_moves:
            moves = ok_moves
        # 网络推理
        p, v = self.net.predict(board)
        # 只保留合法走法的概率，重归一化
        probs = {}
        total = 0.0
        for mv in moves:
            prob = float(p[mv])
            probs[mv] = prob
            total += prob
        if total <= 0:
            for mv in moves:
                probs[mv] = 1.0 / len(moves)
        else:
            for mv in moves:
                probs[mv] /= total
        for mv in moves:
            node.children[mv] = MCTSNode(probs[mv], parent=node, move=mv)

    def _backup(self, node: MCTSNode, value: float):
        """反向传播（value 是当前节点视角）"""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # 父节点视角取反
            node = node.parent

    def _simulate(self, board: XiangqiBoard, root: MCTSNode):
        """一次模拟：select -> expand -> evaluate -> backup（用 make/unmake 推进棋盘）"""
        node = root
        path = []
        # 沿树下降
        while node.is_expanded() and node.children:
            best_move = None
            best_val = -float('inf')
            for mv, child in node.children.items():
                val = self._uct(node, child)
                if val > best_val:
                    best_val = val
                    best_move = mv
            node = node.children[best_move]
            board.make_move(best_move)
            path.append(node)
        # 到达叶节点
        if not node.is_expanded():
            self._expand(board, node)
        # 评估：若游戏结束则用真实结果，否则用网络价值
        result = board.result()
        if result is not None:
            if result == '1-0':
                value = 1.0 if board.turn == BLACK else -1.0  # 红胜，当前方（board.turn）视角
            elif result == '0-1':
                value = 1.0 if board.turn == RED else -1.0
            else:
                value = 0.0
        else:
            _, v = self.net.predict(board)
            value = v
        # 回退路径（从叶节点开始）
        node.visit_count += 1
        node.total_value += value
        value = -value
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value
            value = -value
        # 撤销走法
        for _ in path:
            board.unmake_move()

    def search(self, board: XiangqiBoard, temperature: float = 1.0) -> dict:
        """在根节点执行 num_simulations 次模拟，返回走法概率分布 {move: prob}"""
        root = MCTSNode(0.0)
        self._expand(board, root)
        if not root.children:
            # 无合法走法（被将死或困毙），返回空分布
            return {}
        # 根节点加 Dirichlet 噪声（训练时探索）
        if self.dirichlet_eps > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((len(root.children),), self.dirichlet_alpha)).sample()
            for i, (mv, child) in enumerate(root.children.items()):
                child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * float(noise[i])
        for _ in range(self.num_simulations):
            self._simulate(board, root)
        # 计算根节点概率
        if temperature == 0:
            # 贪心：选访问次数最多的
            best_mv = max(root.children, key=lambda m: root.children[m].visit_count)
            return {best_mv: 1.0}
        visits = {mv: child.visit_count for mv, child in root.children.items()}
        if temperature == 1.0:
            total = sum(visits.values())
            return {mv: v / total for mv, v in visits.items()}
        # 温度调度
        exp_visits = {mv: v ** (1.0 / temperature) for mv, v in visits.items()}
        total = sum(exp_visits.values())
        return {mv: v / total for mv, v in exp_visits.items()}

    def best_move(self, board: XiangqiBoard, temperature: float = 0.0) -> int:
        """返回最佳走法；无合法走法（被将死或困毙）时返回 -1"""
        probs = self.search(board, temperature)
        if not probs:
            return -1
        return max(probs, key=probs.get)


# ---------------------------------------------------------------------------
# 批量推理池架构（Leela Chess Zero 式）：
#   Evaluator 抽象 + DirectEvaluator（本地）/ ServerEvaluator（推理服务）
#   + BatchMCTS（批式搜索）。原 MCTS 保留不动，供 GUI 人机对战继续使用。
# ---------------------------------------------------------------------------

class Evaluator:
    """推理评估抽象：evaluate(boards) -> (ps, vs)

    - ps: list[list[float]]，每个局面的 8100 维策略概率
    - vs: list[float]，每个局面的价值（当前方视角）
    两个实现均返回纯 Python 数据，BatchMCTS 无需区分本地/远程。
    """

    def evaluate(self, boards):
        raise NotImplementedError


class DirectEvaluator(Evaluator):
    """本地直接推理评估器：进程内持有网络副本，直接 predict_batch"""

    def __init__(self, net: XiangqiNet):
        self.net = net

    def evaluate(self, boards):
        if not boards:
            return [], []
        p, v = self.net.predict_batch(boards)  # 返回 CPU 张量（内部已对齐 device）
        return p.tolist(), v.tolist()


class ServerEvaluator(Evaluator):
    """远程推理服务评估器：发请求到 InferenceServer，阻塞等待响应

    每个 worker 拥有独立响应队列（resp_queue），同一队列内按请求顺序
    返回（FIFO），因此 req_id 自增配对天然成立，多 worker 无交叉消费。
    """

    def __init__(self, req_queue, resp_queue, worker_id: int):
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.worker_id = worker_id
        self._next_req_id = 0

    def evaluate(self, boards):
        if not boards:
            return [], []
        req_id = self._next_req_id
        self._next_req_id += 1
        self.req_queue.put({
            'worker_id': self.worker_id,
            'req_id': req_id,
            'boards': boards,
        })
        resp = self.resp_queue.get()  # 阻塞等待本 worker 队列的响应
        return resp['ps'], resp['vs']


class BatchMCTS:
    """批式 MCTS：select 阶段纯树操作不调用网络，叶局面攒批统一评估

    流程：
      (a) 根节点先验并入第一批评估；
      (b) 执行 num_simulations 次 select，收集到达的未扩展叶局面
          （可哈希局面标识 fen 去重），记录每条模拟路径与对应叶；
      (c) 所有唯一叶局面一次性 evaluate；
      (d) 用返回的 p 批量扩展叶子，v 按路径 backup。
    搜索语义与原 MCTS 等价（UBC、Dirichlet 噪声、温度采样不变）。
    """

    def __init__(self, evaluator: Evaluator, c_puct: float = 1.5,
                 num_simulations: int = 200, virtual_loss: int = 1,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 batch_size: int = 64):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.virtual_loss = virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        # 每轮批量收集的模拟条数：收集 batch_size 条路径 -> 批量评估唯一叶 ->
        # 统一扩展 + backup，再进入下一轮，使 UCB 统计随轮次更新。
        # batch_size=1 时与逐模拟扩展的 MCTS 完全等价（语义一致性基准）。
        self.batch_size = max(1, int(batch_size))

    # ---- 与 MCTS 相同的树操作 ----
    def _uct(self, node: MCTSNode, child: MCTSNode) -> float:
        q = -child.q_value  # 子节点价值是对方视角，取反
        u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
        return q + u

    def _expand_node(self, node: MCTSNode, board: XiangqiBoard, p_list):
        """用网络先验 p_list 扩展节点（含棋规长将/长捉过滤）"""
        moves = board.legal_moves()
        node.legal_moves = moves
        if not moves:
            return
        ok_moves = [mv for mv in moves if not would_be_illegal_cycle(board, mv)]
        if ok_moves:
            moves = ok_moves
        total = 0.0
        probs = {}
        for mv in moves:
            prob = float(p_list[mv])
            probs[mv] = prob
            total += prob
        if total <= 0:
            for mv in moves:
                probs[mv] = 1.0 / len(moves)
        else:
            for mv in moves:
                probs[mv] /= total
        for mv in moves:
            node.children[mv] = MCTSNode(probs[mv], parent=node, move=mv)

    def _terminal_value(self, board: XiangqiBoard) -> float:
        """游戏结束局面的真实结果价值（board.turn 视角）"""
        result = board.result()
        if result == '1-0':
            return 1.0 if board.turn == BLACK else -1.0
        if result == '0-1':
            return 1.0 if board.turn == RED else -1.0
        return 0.0  # '1/2-1/2' 或 None

    def search(self, board: XiangqiBoard, temperature: float = 1.0) -> dict:
        """批式搜索：返回走法概率分布 {move: prob}"""
        root = MCTSNode(0.0)
        # (a) 根节点先验（并入第一批评估）
        root_p, _ = self.evaluator.evaluate([board])
        self._expand_node(root, board, root_p[0])
        if not root.children:
            return {}
        # 根节点加 Dirichlet 噪声（训练时探索）
        if self.dirichlet_eps > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((len(root.children),), self.dirichlet_alpha)).sample()
            for i, (mv, child) in enumerate(root.children.items()):
                child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * float(noise[i])
        # (b) 分轮批量收集：每轮收集 batch_size 条模拟路径（不足则到
        # num_simulations 为止），收集完统一评估扩展后再进下一轮，
        # 保证后续 select 能看到已更新的 visit/prior 统计（UCB 正常工作）
        sims_done = 0
        while sims_done < self.num_simulations:
            batch = []  # (path, leaf_node, leaf_board_snapshot)
            leaf_boards = {}  # fen -> board snapshot（可哈希局面标识去重）
            while len(batch) < self.batch_size and sims_done < self.num_simulations:
                node = root
                path = []
                b = board.copy()  # 从根快照独立推进，不改动调用方棋盘
                while node.is_expanded() and node.children:
                    best_move = max(node.children, key=lambda m: self._uct(node, node.children[m]))
                    child = node.children[best_move]
                    path.append(child)
                    node = child
                    b.make_move(child.move)
                # 到达未扩展叶（含游戏结束局面）
                sims_done += 1
                batch.append((path, node, b))
                leaf_boards[b.fen()] = b
            # (c) 本轮唯一叶局面批量评估
            unique_boards = list(leaf_boards.values())
            if unique_boards:
                ps, vs = self.evaluator.evaluate(unique_boards)
                eval_by_fen = {b.fen(): (ps[i], vs[i]) for i, b in enumerate(unique_boards)}
            else:
                eval_by_fen = {}
            # (d) 批量扩展 + 按路径 backup
            for path, node, b in batch:
                result = b.result()
                if result is not None:
                    # 游戏结束：用真实结果，不扩展不推理
                    value = self._terminal_value(b)
                else:
                    p_leaf, v_leaf = eval_by_fen[b.fen()]
                    value = v_leaf
                    if not node.is_expanded():
                        self._expand_node(node, b, p_leaf)
                # 反向传播（value 是当前节点视角）
                node.visit_count += 1
                node.total_value += value
                value = -value
                for n in reversed(path):
                    n.visit_count += 1
                    n.total_value += value
                    value = -value
        # 根节点概率分布（与原 MCTS 相同：温度调度）
        if temperature == 0:
            best_mv = max(root.children, key=lambda m: root.children[m].visit_count)
            return {best_mv: 1.0}
        visits = {mv: child.visit_count for mv, child in root.children.items()}
        if temperature == 1.0:
            total = sum(visits.values())
            return {mv: v / total for mv, v in visits.items()}
        exp_visits = {mv: v ** (1.0 / temperature) for mv, v in visits.items()}
        total = sum(exp_visits.values())
        return {mv: v / total for mv, v in exp_visits.items()}

    def best_move(self, board: XiangqiBoard, temperature: float = 0.0) -> int:
        """返回最佳走法；无合法走法（被将死或困毙）时返回 -1"""
        probs = self.search(board, temperature)
        if not probs:
            return -1
        return max(probs, key=probs.get)
