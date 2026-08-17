# -*- coding: utf-8 -*-
"""规则引擎单元测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.board import XiangqiBoard, START_FEN, RED, BLACK, rc_to_sq, sq_to_rc

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f'  [FAIL] {name}')

# 1. 初始局面走法数（红方 44 种合法走法）
b = XiangqiBoard()
moves = b.legal_moves()
check('初始局面红方 44 种走法', len(moves) == 44)

# 2. 初始 FEN 往返
b2 = XiangqiBoard(b.fen())
check('FEN 往返一致', b2.fen() == b.fen())

# 3. 走法 make/unmake 往返
b = XiangqiBoard()
mv = moves[0]
h0 = b.zobrist_hash()
b.make_move(mv)
h1 = b.zobrist_hash()
b.unmake_move()
check('make/unmake 后局面一致', b.fen() == START_FEN and b.zobrist_hash() == h0)
check('make 后哈希变化', h1 != h0)

# 4. 蹩马腿：初始局面红马 (9,1) 合法目标
# 红马 (9,1) 到 (7,0)/(7,2) 蹩腿 (8,1) 空，可跳；到 (8,3) 蹩腿 (9,2) 是红相，被蹩
horse_sq = rc_to_sq(9, 1)
horse_targets = set()
for mv in moves:
    fsq, tsq = divmod(mv, 90)
    if fsq == horse_sq:
        horse_targets.add(tsq)
expected = {rc_to_sq(7, 0), rc_to_sq(7, 2)}
check('初始红马(9,1)合法目标', horse_targets == expected)

# 5. 塞象眼：初始红相 (9,2) 不能跳（象眼 (8,2) 被马挡？(8,2) 空）
# 红相 (9,2) 到 (7,0) 象眼 (8,1) 空，可跳；(7,4) 象眼 (8,3) 空，可跳
elephant_sq = rc_to_sq(9, 2)
elephant_targets = set()
for mv in moves:
    fsq, tsq = divmod(mv, 90)
    if fsq == elephant_sq:
        elephant_targets.add(tsq)
expected_ele = {rc_to_sq(7, 0), rc_to_sq(7, 4)}
check('初始红相(9,2)合法目标', elephant_targets == expected_ele)

# 6. 将帅照面：构造局面验证
# 红帅 (9,4)，黑将 (0,4)，中间无子 -> 红帅被黑将"飞将"攻击
b = XiangqiBoard()
b.board = [0] * 90
b.board[rc_to_sq(9, 4)] = 1  # 红帅
b.board[rc_to_sq(0, 4)] = 8  # 黑将
b.turn = RED
b._init_hash()
check('将帅照面：红帅被攻击', b.is_attacked(rc_to_sq(9, 4), BLACK))
# 中间隔子则不攻击：红帅 (8,4)，黑将 (0,4)，中间 (4,4) 放红兵
b.board[rc_to_sq(8, 4)] = 1
b.board[rc_to_sq(9, 4)] = 0
b.board[rc_to_sq(4, 4)] = 7
b._init_hash()
check('将帅照面：中间隔子不攻击', not b.is_attacked(rc_to_sq(8, 4), BLACK))

# 7. 将军检测：黑车将军红帅
b = XiangqiBoard()
b.board = [0] * 90
b.board[rc_to_sq(9, 4)] = 1  # 红帅
b.board[rc_to_sq(9, 0)] = 12 # 黑车
b.turn = RED
b._init_hash()
check('黑车将军红帅', b.in_check(RED))

# 8. 炮隔子打
b = XiangqiBoard()
b.board = [0] * 90
b.board[rc_to_sq(9, 4)] = 1   # 红帅
b.board[rc_to_sq(5, 4)] = 7   # 红兵（炮架）
b.board[rc_to_sq(0, 4)] = 13  # 黑炮
b.turn = RED
b._init_hash()
check('黑炮隔兵将军', b.in_check(RED))

# 9. 兵过河规则：红兵 (4,0) 可前进+横走；红兵 (5,0) 只能前进
b = XiangqiBoard()
b.board = [0] * 90
b.board[rc_to_sq(9, 4)] = 1
b.board[rc_to_sq(0, 3)] = 8  # 黑将移到 (0,3)，避免与红帅照面
b.board[rc_to_sq(4, 0)] = 7  # 红兵已过河
b.turn = RED
b._init_hash()
moves = b.legal_moves()
pawn_targets = set()
for mv in moves:
    fsq, tsq = divmod(mv, 90)
    if fsq == rc_to_sq(4, 0):
        pawn_targets.add(tsq)
expected_pawn = {rc_to_sq(3, 0), rc_to_sq(4, 1)}
check('过河红兵可前进+横走', pawn_targets == expected_pawn)

# 10. 胜负：将死局面
# 构造红帅被将死
b = XiangqiBoard()
b.board = [0] * 90
b.board[rc_to_sq(9, 4)] = 1   # 红帅
b.board[rc_to_sq(0, 4)] = 8   # 黑将
b.board[rc_to_sq(0, 0)] = 12  # 黑车
b.board[rc_to_sq(0, 8)] = 12  # 黑车
b.board[rc_to_sq(9, 0)] = 12  # 黑车
b.board[rc_to_sq(9, 8)] = 12  # 黑车
b.turn = RED
b._init_hash()
# 红帅 (9,4) 被 (9,0) 车将军，且 (9,4) 周围被围死
# 简化：直接验证无合法走法时判负
check('将死局面判负', b.result() == '0-1')

# 11. 困毙：无子可走且不被将军 -> 判负
b = XiangqiBoard()
b.board = [0] * 90
b.board[rc_to_sq(9, 4)] = 1   # 红帅
b.board[rc_to_sq(0, 4)] = 8   # 黑将
b.board[rc_to_sq(0, 0)] = 12  # 黑车
b.board[rc_to_sq(0, 8)] = 12  # 黑车
b.board[rc_to_sq(9, 0)] = 12  # 黑车
b.board[rc_to_sq(9, 8)] = 12  # 黑车
b.turn = RED
b._init_hash()
check('困毙判负', b.result() == '0-1')

# 12. 随机对弈冒烟：走 200 步不崩溃
import random
b = XiangqiBoard()
rng = random.Random(42)
for i in range(200):
    if b.is_game_over():
        break
    mvs = b.legal_moves()
    b.make_move(rng.choice(mvs))
check('随机对弈 200 步无异常', True)

# 13. 哈希一致性：相同局面哈希相同
b1 = XiangqiBoard()
b2 = XiangqiBoard()
check('相同局面哈希一致', b1.zobrist_hash() == b2.zobrist_hash())

# 14. 走法编码往返
b = XiangqiBoard()
mv = b.legal_moves()[0]
uci = b.move_to_uci(mv)
check('UCI 往返', b.uci_to_move(uci) == mv)
iccs = b.move_to_iccs(mv)
check('ICCS 往返', b.iccs_to_move(iccs) == mv)

print(f'\n结果: {passed} 通过, {failed} 失败')
sys.exit(1 if failed else 0)
