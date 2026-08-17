# -*- coding: utf-8 -*-
"""经典评估函数：子力价值 + 位置活性 + 机动性"""
from core.board import (
    XiangqiBoard, RED, BLACK, EMPTY,
    R_KING, R_ADVISOR, R_ELEPHANT, R_HORSE, R_ROOK, R_CANNON, R_PAWN,
    B_KING, B_ADVISOR, B_ELEPHANT, B_HORSE, B_ROOK, B_CANNON, B_PAWN,
    rc_to_sq, sq_to_rc,
)

# 子力基础价值
PIECE_VALUE = {
    R_KING: 100000, R_ADVISOR: 200, R_ELEPHANT: 200, R_HORSE: 450,
    R_ROOK: 1000, R_CANNON: 450, R_PAWN: 200,
    B_KING: 100000, B_ADVISOR: 200, B_ELEPHANT: 200, B_HORSE: 450,
    B_ROOK: 1000, B_CANNON: 450, B_PAWN: 200,
}

# 位置价值表（红方视角，row 0 在上=黑方，row 9 在下=红方）
# 马的位置价值（中心活跃）
HORSE_POS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# 简化位置表：马（红方视角，值越大越好）
HORSE_POS_RED = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# 兵位置价值（红方视角，越深入对方阵地价值越高）
PAWN_POS_RED = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# 用程序生成位置表
def _build_pos_tables():
    # 马：中心位置价值高
    horse = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # 马在河界附近（row 3-6）价值高
    for r in range(3, 7):
        for c in range(1, 8):
            horse[r][c] = 20
    for r in range(4, 6):
        for c in range(2, 7):
            horse[r][c] = 30
    horse[4][3] = horse[4][4] = horse[4][5] = 40
    horse[5][3] = horse[5][4] = horse[5][5] = 40

    # 兵：过河后价值递增
    pawn = [[0] * 9 for _ in range(10)]
    for c in range(9):
        pawn[4][c] = 30   # 刚过河
        pawn[3][c] = 50
        pawn[2][c] = 60
        pawn[1][c] = 70
        pawn[0][c] = 80   # 深入敌阵
    # 中路兵价值略高
    for r in range(0, 5):
        pawn[r][4] += 10

    # 炮：中路和河界附近价值高
    cannon = [[0] * 9 for _ in range(10)]
    for r in range(2, 8):
        for c in range(1, 8):
            cannon[r][c] = 10
    cannon[4][4] = 20
    cannon[5][4] = 20

    # 车：全盘价值均匀，中路略高
    rook = [[0] * 9 for _ in range(10)]
    for r in range(10):
        for c in range(9):
            if 2 <= c <= 6:
                rook[r][c] = 5

    return horse, pawn, cannon, rook

HORSE_POS, PAWN_POS, CANNON_POS, ROOK_POS = _build_pos_tables()


def _pos_value(piece, r, c):
    """位置价值（红方视角）"""
    if piece == R_HORSE:
        return HORSE_POS[r][c]
    if piece == R_PAWN:
        return PAWN_POS[r][c]
    if piece == R_CANNON:
        return CANNON_POS[r][c]
    if piece == R_ROOK:
        return ROOK_POS[r][c]
    return 0


def evaluate(board: XiangqiBoard) -> float:
    """返回红方视角的评估值（>0 红优，<0 黑优）"""
    score = 0
    for sq, p in enumerate(board.board):
        if p == EMPTY:
            continue
        r, c = sq_to_rc(sq)
        if p <= 7:  # 红方
            score += PIECE_VALUE[p] + _pos_value(p, r, c)
        else:       # 黑方（镜像位置）
            score -= PIECE_VALUE[p] + _pos_value(p, 9 - r, c)
    return score


def evaluate_side(board: XiangqiBoard, color: int) -> float:
    """返回指定方视角的评估值"""
    v = evaluate(board)
    return v if color == RED else -v
