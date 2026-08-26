# -*- coding: utf-8 -*-
"""经典 AI：alpha-beta 剪枝 + 迭代加深 + 走法排序"""
import time
from core.board import XiangqiBoard, RED, BLACK
from ai.evaluator import evaluate, evaluate_side
from core.rules import would_be_illegal_cycle, is_repetition, classify_move

# 走法排序：吃子优先（MVV-LVA），将军优先
PIECE_ORDER = {1: 6, 2: 1, 3: 1, 4: 3, 5: 5, 6: 3, 7: 2,
               8: 6, 9: 1, 10: 1, 11: 3, 12: 5, 13: 3, 14: 2}


def _move_score(board: XiangqiBoard, mv: int) -> int:
    fsq, tsq = divmod(mv, 90)
    captured = board.board[tsq]
    score = 0
    if captured:
        # MVV-LVA：吃大子优先
        score = 10000 + PIECE_ORDER[captured] * 100 - PIECE_ORDER[board.board[fsq]]
    # 棋规约束：走法导致局面重复且为将/捉/杀 → 降权（避免长将/长捉/长杀）
    if is_repetition(board, mv):
        t = classify_move(board, mv)
        if t in ('将', '捉', '杀'):
            score -= 5000
    return score


class ClassicAI:
    """alpha-beta 搜索 AI，支持迭代加深与时间/深度限制"""

    def __init__(self, depth: int = 3, time_limit: float = 1.0):
        self.depth = depth
        self.time_limit = time_limit
        self.nodes = 0
        self._deadline = 0.0

    def _order_moves(self, board: XiangqiBoard, moves: list) -> list:
        scored = [(_move_score(board, m), m) for m in moves]
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    def _search(self, board: XiangqiBoard, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if time.time() > self._deadline:
            raise TimeoutError
        result = board.result()
        if result is not None:
            if result == '1-0':
                return 1000000 + depth
            if result == '0-1':
                return -1000000 - depth
            return 0
        if depth == 0:
            return evaluate(board)
        moves = self._order_moves(board, board.legal_moves())
        if board.turn == RED:
            best = -float('inf')
            for mv in moves:
                board.make_move(mv)
                v = self._search(board, depth - 1, alpha, beta)
                board.unmake_move()
                if v > best:
                    best = v
                if best > alpha:
                    alpha = best
                if alpha >= beta:
                    break
            return best
        else:
            best = float('inf')
            for mv in moves:
                board.make_move(mv)
                v = self._search(board, depth - 1, alpha, beta)
                board.unmake_move()
                if v < best:
                    best = v
                if best < beta:
                    beta = best
                if alpha >= beta:
                    break
            return best

    def best_move(self, board: XiangqiBoard) -> int:
        """返回最佳走法（迭代加深）。搜索在副本上进行，不修改原棋盘。"""
        moves = board.legal_moves()
        if not moves:
            return -1
        if len(moves) == 1:
            return moves[0]
        work = board.copy()
        # 棋规约束：剔除会导致自己长将/长捉/长杀违规的走法
        ok_moves = []
        for mv in moves:
            if not would_be_illegal_cycle(work, mv):
                ok_moves.append(mv)
        if ok_moves:
            moves = ok_moves
        self._deadline = time.time() + self.time_limit
        best = moves[0]
        try:
            for d in range(1, self.depth + 1):
                alpha = -float('inf')
                beta = float('inf')
                ordered = self._order_moves(work, moves)
                cur_best = ordered[0]
                for mv in ordered:
                    work.make_move(mv)
                    v = self._search(work, d - 1, alpha, beta)
                    work.unmake_move()
                    if board.turn == RED:
                        if v > alpha:
                            alpha = v
                            cur_best = mv
                    else:
                        if v < beta:
                            beta = v
                            cur_best = mv
                best = cur_best
        except TimeoutError:
            pass
        return best


def make_classic_ai(difficulty: str) -> ClassicAI:
    """难度分级：easy=1层, medium=2层, hard=3层, master=4层"""
    depth_map = {'easy': 1, 'medium': 2, 'hard': 3, 'master': 4}
    time_map = {'easy': 0.3, 'medium': 0.8, 'hard': 1.5, 'master': 3.0}
    d = depth_map.get(difficulty, 3)
    t = time_map.get(difficulty, 1.5)
    return ClassicAI(depth=d, time_limit=t)
