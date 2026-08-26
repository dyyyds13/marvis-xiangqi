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
