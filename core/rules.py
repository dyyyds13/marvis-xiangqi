# -*- coding: utf-8 -*-
"""中国象棋棋规（2011版）循环局面判定模块

核心能力：
1. 走法性质分类：将 / 杀 / 捉 / 闲（兑、献归为闲）
2. 循环局面检测（基于 Zobrist 哈希历史）
3. 循环裁决：长将 / 长捉 / 长杀判负、双方循环作和、一将一捉等判负
4. AI 约束辅助：判断某走法是否会导致自己长将/长捉违规

"捉"的完整判定（方案A）：
- 走子后攻击对方子，对方子无根（无保护）→ 捉
- 对方子有根，但攻击子价值 < 被捉子价值（用小子换大子）→ 捉
- 对方子有根，攻击子价值 >= 被捉子价值（等价交换）→ 兑（算闲）
"""
from __future__ import annotations

from .board import (
    EMPTY, RED, BLACK, SQUARES, ROWS, COLS,
    R_KING, R_ADVISOR, R_ELEPHANT, R_HORSE, R_ROOK, R_CANNON, R_PAWN,
    B_KING, B_ADVISOR, B_ELEPHANT, B_HORSE, B_ROOK, B_CANNON, B_PAWN,
    color_of, sq_to_rc, rc_to_sq, in_board,
    RED_PALACE_SET, BLACK_PALACE_SET,
    HORSE_MOVES, ELEPHANT_MOVES, ADVISOR_MOVES, KING_MOVES, ROOK_DIRS,
)

# ---------- 走法性质 ----------
JIANG = '将'    # 将军
SHA = '杀'      # 叫杀
ZHUO = '捉'     # 捉子
XIAN = '闲'     # 闲（含兑、献）

# ---------- 循环判定结果 ----------
DRAW = '1/2-1/2'
RED_WIN = '1-0'
BLACK_WIN = '0-1'

# ---------- 子力价值（用于"捉"的划算判断） ----------
PIECE_VALUE = {
    R_KING: 10000, B_KING: 10000,
    R_ROOK: 900, B_ROOK: 900,
    R_CANNON: 450, B_CANNON: 450,
    R_HORSE: 400, B_HORSE: 400,
    R_ELEPHANT: 200, B_ELEPHANT: 200,
    R_ADVISOR: 200, B_ADVISOR: 200,
    R_PAWN: 100, B_PAWN: 100,
}


# =====================================================================
# 攻击格子计算
# =====================================================================
def attacked_squares(board, sq: int) -> list:
    """sq 上的棋子攻击的格子列表（含可吃子位置，不含己方子过滤）"""
    p = board.board[sq]
    if p == EMPTY:
        return []
    r, c = sq_to_rc(sq)
    color = color_of(p)
    targets = []
    if p in (R_ROOK, B_ROOK):
        for dr, dc in ROOK_DIRS:
            nr, nc = r + dr, c + dc
            while in_board(nr, nc):
                nsq = rc_to_sq(nr, nc)
                targets.append(nsq)
                if board.board[nsq] != EMPTY:
                    break
                nr += dr
                nc += dc
    elif p in (R_CANNON, B_CANNON):
        for dr, dc in ROOK_DIRS:
            nr, nc = r + dr, c + dc
            jumped = False
            while in_board(nr, nc):
                nsq = rc_to_sq(nr, nc)
                if not jumped:
                    if board.board[nsq] == EMPTY:
                        targets.append(nsq)
                    else:
                        jumped = True
                else:
                    if board.board[nsq] != EMPTY:
                        targets.append(nsq)
                        break
                nr += dr
                nc += dc
    elif p in (R_HORSE, B_HORSE):
        for (dr, dc), (br, bc) in HORSE_MOVES:
            nr, nc = r + dr, c + dc
            if not in_board(nr, nc):
                continue
            leg = rc_to_sq(r + br, c + bc)
            if board.board[leg] != EMPTY:
                continue
            targets.append(rc_to_sq(nr, nc))
    elif p in (R_PAWN, B_PAWN):
        if p == R_PAWN:
            dirs = [(-1, 0)]
            if r <= 4:  # 红兵过河后可横走
                dirs += [(0, -1), (0, 1)]
        else:
            dirs = [(1, 0)]
            if r >= 5:  # 黑卒过河后可横走
                dirs += [(0, -1), (0, 1)]
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_board(nr, nc):
                targets.append(rc_to_sq(nr, nc))
    elif p in (R_KING, B_KING):
        palace = RED_PALACE_SET if p == R_KING else BLACK_PALACE_SET
        for dr, dc in KING_MOVES:
            nr, nc = r + dr, c + dc
            if in_board(nr, nc):
                nsq = rc_to_sq(nr, nc)
                if nsq in palace:
                    targets.append(nsq)
        # 飞将（照面）：同列且中间无子
        kpos = board._king_pos[1 - color]
        if kpos >= 0:
            kr, kc = sq_to_rc(kpos)
            if kc == c:
                lo, hi = min(r, kr), max(r, kr)
                blocked = False
                for rr in range(lo + 1, hi):
                    if board.board[rc_to_sq(rr, c)] != EMPTY:
                        blocked = True
                        break
                if not blocked:
                    targets.append(kpos)
    elif p in (R_ADVISOR, B_ADVISOR):
        palace = RED_PALACE_SET if p == R_ADVISOR else BLACK_PALACE_SET
        for dr, dc in ADVISOR_MOVES:
            nr, nc = r + dr, c + dc
            if in_board(nr, nc):
                nsq = rc_to_sq(nr, nc)
                if nsq in palace:
                    targets.append(nsq)
    elif p in (R_ELEPHANT, B_ELEPHANT):
        for (dr, dc), (er, ec) in ELEPHANT_MOVES:
            nr, nc = r + dr, c + dc
            if not in_board(nr, nc):
                continue
            eye = rc_to_sq(r + er, c + ec)
            if board.board[eye] != EMPTY:
                continue
            nsq = rc_to_sq(nr, nc)
            if p == R_ELEPHANT and nr >= 5:
                targets.append(nsq)
            elif p == B_ELEPHANT and nr <= 4:
                targets.append(nsq)
    return targets


# =====================================================================
# "捉"的判定
# =====================================================================
def _has_root(board, sq: int, color: int) -> bool:
    """sq 上的 color 方棋子是否有根（被己方其他子保护）"""
    for s, p in enumerate(board.board):
        if p == EMPTY or color_of(p) != color:
            continue
        if s == sq:
            continue
        if sq in attacked_squares(board, s):
            return True
    return False


def _is_capture_worthwhile(board, attacker_sq: int, target_sq: int) -> bool:
    """吃子是否划算：无根子或小子换大子 → 捉；等价交换 → 兑（不算捉）"""
    attacker = board.board[attacker_sq]
    target = board.board[target_sq]
    target_color = color_of(target)
    if not _has_root(board, target_sq, target_color):
        return True  # 捉无根子
    if PIECE_VALUE[attacker] < PIECE_VALUE[target]:
        return True  # 用小子换大子
    return False  # 等价交换 → 兑


def _is_capture_threat(board, color: int) -> bool:
    """color 方走子后是否形成捉（攻击对方无根子或用小子换大子）"""
    opp = 1 - color
    for sq, p in enumerate(board.board):
        if p == EMPTY or color_of(p) != color:
            continue
        for target in attacked_squares(board, sq):
            tp = board.board[target]
            if tp == EMPTY or color_of(tp) != opp:
                continue
            if _is_capture_worthwhile(board, sq, target):
                return True
    return False


# =====================================================================
# "杀"的判定
# =====================================================================
def _is_shai(board, color: int) -> bool:
    """color 方走子后，对方是否被叫杀（对方无论怎么走，下一步都被将死）"""
    opp = 1 - color
    opp_moves = board.legal_moves()
    if not opp_moves:
        return True  # 已被将死
    for mv in opp_moves:
        board.make_move(mv)
        can_mate = False
        for mv2 in board.legal_moves():
            board.make_move(mv2)
            if not board.legal_moves():
                can_mate = True
            board.unmake_move()
            if can_mate:
                break
        board.unmake_move()
        if not can_mate:
            return False
    return True


# =====================================================================
# 走法性质分类
# =====================================================================
def classify_move(board, mv: int) -> str:
    """分类走法：'将' / '杀' / '捉' / '闲'（兑、献归为闲）"""
    mover = board.turn
    board.make_move(mv)
    try:
        opp = 1 - mover
        # 将：走完后将军对方
        if board.in_check(opp):
            return JIANG
        # 杀：走完后对方被将死或叫杀
        if not board.legal_moves():
            return SHA
        if _is_shai(board, mover):
            return SHA
        # 捉：走完后攻击对方无根子或用小子换大子
        if _is_capture_threat(board, mover):
            return ZHUO
        return XIAN
    finally:
        board.unmake_move()


# =====================================================================
# 循环检测与裁决
# =====================================================================
def _classify_sequence(types: list) -> str:
    """分析一个走子方的走法序列性质（长将/长捉/长杀/一将一捉/.../闲）"""
    if not types:
        return '闲'
    if all(t == JIANG for t in types):
        return '长将'
    if all(t == ZHUO for t in types):
        return '长捉'
    if all(t == SHA for t in types):
        return '长杀'
    has_j = JIANG in types
    has_z = ZHUO in types
    has_s = SHA in types
    if has_j and has_z:
        return '一将一捉'
    if has_j and has_s:
        return '一将一杀'
    if has_z and has_s:
        return '一捉一杀'
    if has_j:
        return '一将一闲'
    if has_z:
        return '一捉一闲'
    if has_s:
        return '一杀一闲'
    return '闲'


def _judge_cycle(red_types: list, black_types: list) -> str:
    """按棋规（2011）裁决循环，返回 '1-0'/'0-1'/'1/2-1/2'"""
    red_seq = _classify_sequence(red_types)
    black_seq = _classify_sequence(black_types)
    # 长将最严重：长将方判负（双方长将则和）
    if red_seq == '长将' and black_seq == '长将':
        return DRAW
    if red_seq == '长将':
        return BLACK_WIN
    if black_seq == '长将':
        return RED_WIN
    # 长捉/长杀
    if red_seq == '长捉' and black_seq == '长捉':
        return DRAW
    if red_seq == '长杀' and black_seq == '长杀':
        return DRAW
    if red_seq == '长捉' and black_seq == '长杀':
        return BLACK_WIN  # 长捉比长杀严重
    if red_seq == '长杀' and black_seq == '长捉':
        return RED_WIN
    if red_seq == '长捉':
        return BLACK_WIN
    if red_seq == '长杀':
        return BLACK_WIN
    if black_seq == '长捉':
        return RED_WIN
    if black_seq == '长杀':
        return RED_WIN
    # 一将一捉 / 一将一杀 / 一捉一杀 → 判负
    if red_seq in ('一将一捉', '一将一杀', '一捉一杀'):
        return BLACK_WIN
    if black_seq in ('一将一捉', '一将一杀', '一捉一杀'):
        return RED_WIN
    # 双方都闲 / 一将一闲 / 一捉一闲 → 和
    return DRAW


def _analyze_cycle(board, cycle_moves: list):
    """分析循环周期中每步棋的性质，返回 (red_types, black_types)"""
    work = board.copy()
    for _ in range(len(cycle_moves)):
        work.unmake_move()
    red_types = []
    black_types = []
    for entry in cycle_moves:
        fsq, tsq = entry[0], entry[1]
        mv = fsq * 90 + tsq
        mover = work.turn
        t = classify_move(work, mv)
        if mover == RED:
            red_types.append(t)
        else:
            black_types.append(t)
        work.make_move(mv)
    return red_types, black_types


def check_cycle(board):
    """检测循环局面并按棋规裁决，返回 '1-0'/'0-1'/'1/2-1/2'/None"""
    h = board._hash
    hist = board._hash_history
    if hist.count(h) < 3:
        return None
    first_idx = hist.index(h)
    cycle_moves = board._history[first_idx:]
    if len(cycle_moves) < 2:
        return None
    red_types, black_types = _analyze_cycle(board, cycle_moves)
    return _judge_cycle(red_types, black_types)


# =====================================================================
# AI 约束辅助
# =====================================================================
def would_be_illegal_cycle(board, mv: int) -> bool:
    """走 mv 后是否会导致自己长将/长捉/长杀违规（AI 应避免）"""
    mover = board.turn
    board.make_move(mv)
    try:
        result = check_cycle(board)
        if result is None:
            return False
        # 循环裁决判自己负 → 违规
        if result == (BLACK_WIN if mover == RED else RED_WIN):
            return True
        return False
    finally:
        board.unmake_move()


def is_repetition(board, mv: int) -> bool:
    """走 mv 后是否形成局面重复（可能进入循环，AI 可据此降权）"""
    board.make_move(mv)
    try:
        return board._hash_history.count(board._hash) >= 2
    finally:
        board.unmake_move()
