# -*- coding: utf-8 -*-
"""中国象棋规则引擎（自研，纯 Python 高效实现）

棋盘：9 列 x 10 行，索引 sq = row * 9 + col
row 0 = 黑方底线（上方），row 9 = 红方底线（下方）
红方先手。
"""
from __future__ import annotations

# ---------- 棋子编码 ----------
EMPTY = 0
R_KING = 1      # 帅
R_ADVISOR = 2   # 仕
R_ELEPHANT = 3  # 相
R_HORSE = 4     # 马
R_ROOK = 5      # 车
R_CANNON = 6    # 炮
R_PAWN = 7      # 兵
B_KING = 8      # 将
B_ADVISOR = 9   # 士
B_ELEPHANT = 10 # 象
B_HORSE = 11    # 马
B_ROOK = 12     # 车
B_CANNON = 13   # 炮
B_PAWN = 14     # 卒

RED = 0
BLACK = 1

PIECE_CHARS = {
    R_KING: 'K', R_ADVISOR: 'A', R_ELEPHANT: 'B', R_HORSE: 'N',
    R_ROOK: 'R', R_CANNON: 'C', R_PAWN: 'P',
    B_KING: 'k', B_ADVISOR: 'a', B_ELEPHANT: 'b', B_HORSE: 'n',
    B_ROOK: 'r', B_CANNON: 'c', B_PAWN: 'p',
}
CHAR_TO_PIECE = {v: k for k, v in PIECE_CHARS.items()}

# 中文棋子名（GUI 用）
PIECE_NAMES = {
    R_KING: '帅', R_ADVISOR: '仕', R_ELEPHANT: '相', R_HORSE: '马',
    R_ROOK: '车', R_CANNON: '炮', R_PAWN: '兵',
    B_KING: '将', B_ADVISOR: '士', B_ELEPHANT: '象', B_HORSE: '马',
    B_ROOK: '车', B_CANNON: '炮', B_PAWN: '卒',
}

ROWS = 10
COLS = 9
SQUARES = 90

# 初始局面 FEN（红方在下）
START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'


def color_of(piece: int) -> int:
    if piece == EMPTY:
        return -1
    return RED if piece <= 7 else BLACK


def is_red(piece: int) -> bool:
    return 1 <= piece <= 7


def is_black(piece: int) -> bool:
    return 8 <= piece <= 14


def sq_to_rc(sq: int):
    return divmod(sq, COLS)


def rc_to_sq(r: int, c: int) -> int:
    return r * COLS + c


def in_board(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS


# 九宫
RED_PALACE = [(7, 3), (7, 4), (7, 5), (8, 3), (8, 4), (8, 5), (9, 3), (9, 4), (9, 5)]
BLACK_PALACE = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
RED_PALACE_SET = set(rc_to_sq(r, c) for r, c in RED_PALACE)
BLACK_PALACE_SET = set(rc_to_sq(r, c) for r, c in BLACK_PALACE)

# 马走法：((dr, dc), 蹩腿(dr, dc))
HORSE_MOVES = [
    ((-2, -1), (-1, 0)), ((-2, 1), (-1, 0)),
    ((2, -1), (1, 0)), ((2, 1), (1, 0)),
    ((-1, -2), (0, -1)), ((-1, 2), (0, 1)),
    ((1, -2), (0, -1)), ((1, 2), (0, 1)),
]
# 相/象走法：((dr, dc), 象眼(dr//2, dc//2))
ELEPHANT_MOVES = [
    ((-2, -2), (-1, -1)), ((-2, 2), (-1, 1)),
    ((2, -2), (1, -1)), ((2, 2), (1, 1)),
]
# 仕/士走法
ADVISOR_MOVES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
# 帅/将走法
KING_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# 车/炮方向
ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class XiangqiBoard:
    """中国象棋棋盘。走法用整数编码：move = from_sq * 90 + to_sq"""

    def __init__(self, fen: str = START_FEN):
        self.board = [EMPTY] * SQUARES
        self.turn = RED
        self.halfmove = 0      # 无吃子步数（自然限着）
        self.fullmove = 1
        self._history = []     # (from_sq, to_sq, captured, halfmove, turn)
        self._king_pos = [-1, -1]  # [红帅位置, 黑将位置]
        self._zobrist = None
        self._hash = 0
        self._init_zobrist()
        if fen:
            self.set_fen(fen)

    # ---------- Zobrist 哈希 ----------
    def _init_zobrist(self):
        import random
        rng = random.Random(20260814)
        self._zobrist = [[rng.getrandbits(64) for _ in range(15)] for _ in range(SQUARES)]
        self._zobrist_turn = rng.getrandbits(64)

    def _init_hash(self):
        h = 0
        for sq, p in enumerate(self.board):
            if p:
                h ^= self._zobrist[sq][p]
        if self.turn == BLACK:
            h ^= self._zobrist_turn
        self._hash = h

    def zobrist_hash(self) -> int:
        return self._hash

    # ---------- FEN ----------
    def set_fen(self, fen: str):
        parts = fen.split()
        board_part = parts[0]
        rows = board_part.split('/')
        self.board = [EMPTY] * SQUARES
        for r, row in enumerate(rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    self.board[rc_to_sq(r, c)] = CHAR_TO_PIECE[ch]
                    c += 1
        self.turn = RED if parts[1] == 'w' else BLACK
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1
        self._history = []
        self._king_pos = [-1, -1]
        for sq, p in enumerate(self.board):
            if p == R_KING:
                self._king_pos[RED] = sq
            elif p == B_KING:
                self._king_pos[BLACK] = sq
        self._init_hash()

    def fen(self) -> str:
        rows = []
        for r in range(ROWS):
            row = ''
            empty = 0
            for c in range(COLS):
                p = self.board[rc_to_sq(r, c)]
                if p:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += PIECE_CHARS[p]
                else:
                    empty += 1
            if empty:
                row += str(empty)
            rows.append(row)
        turn = 'w' if self.turn == RED else 'b'
        return '/'.join(rows) + f' {turn} - - {self.halfmove} {self.fullmove}'

    # ---------- 基本查询 ----------
    def piece_at(self, sq: int) -> int:
        return self.board[sq]

    def find_king(self, color: int) -> int:
        return self._king_pos[color]

    def _update_king_pos(self, fsq: int, tsq: int, moved: int):
        """make/unmake 后更新将帅位置缓存"""
        if moved == R_KING:
            self._king_pos[RED] = tsq
        elif moved == B_KING:
            self._king_pos[BLACK] = tsq

    def copy(self) -> 'XiangqiBoard':
        b = XiangqiBoard.__new__(XiangqiBoard)
        b.board = self.board[:]
        b.turn = self.turn
        b.halfmove = self.halfmove
        b.fullmove = self.fullmove
        b._history = list(self._history)
        b._king_pos = list(self._king_pos)
        b._zobrist = self._zobrist
        b._zobrist_turn = self._zobrist_turn
        b._hash = self._hash
        return b

    # ---------- 走法生成 ----------
    def _gen_pseudo_moves(self, color: int):
        """生成伪合法走法（不含送将过滤）"""
        moves = []
        for sq, p in enumerate(self.board):
            if p == EMPTY or color_of(p) != color:
                continue
            r, c = sq_to_rc(sq)
            if p in (R_ROOK, B_ROOK):
                for dr, dc in ROOK_DIRS:
                    nr, nc = r + dr, c + dc
                    while in_board(nr, nc):
                        nsq = rc_to_sq(nr, nc)
                        t = self.board[nsq]
                        if t == EMPTY:
                            moves.append(sq * 90 + nsq)
                        else:
                            if color_of(t) != color:
                                moves.append(sq * 90 + nsq)
                            break
                        nr += dr
                        nc += dc
            elif p in (R_CANNON, B_CANNON):
                for dr, dc in ROOK_DIRS:
                    nr, nc = r + dr, c + dc
                    jumped = False
                    while in_board(nr, nc):
                        nsq = rc_to_sq(nr, nc)
                        t = self.board[nsq]
                        if not jumped:
                            if t == EMPTY:
                                moves.append(sq * 90 + nsq)
                            else:
                                jumped = True
                        else:
                            if t != EMPTY:
                                if color_of(t) != color:
                                    moves.append(sq * 90 + nsq)
                                break
                        nr += dr
                        nc += dc
            elif p in (R_HORSE, B_HORSE):
                for (dr, dc), (br, bc) in HORSE_MOVES:
                    nr, nc = r + dr, c + dc
                    if not in_board(nr, nc):
                        continue
                    leg = rc_to_sq(r + br, c + bc)
                    if self.board[leg] != EMPTY:
                        continue
                    nsq = rc_to_sq(nr, nc)
                    t = self.board[nsq]
                    if t == EMPTY or color_of(t) != color:
                        moves.append(sq * 90 + nsq)
            elif p in (R_ELEPHANT, B_ELEPHANT):
                for (dr, dc), (er, ec) in ELEPHANT_MOVES:
                    nr, nc = r + dr, c + dc
                    if not in_board(nr, nc):
                        continue
                    # 不能过河
                    if p == R_ELEPHANT and nr < 5:
                        continue
                    if p == B_ELEPHANT and nr > 4:
                        continue
                    eye = rc_to_sq(r + er, c + ec)
                    if self.board[eye] != EMPTY:
                        continue
                    nsq = rc_to_sq(nr, nc)
                    t = self.board[nsq]
                    if t == EMPTY or color_of(t) != color:
                        moves.append(sq * 90 + nsq)
            elif p in (R_ADVISOR, B_ADVISOR):
                palace = RED_PALACE_SET if p == R_ADVISOR else BLACK_PALACE_SET
                for dr, dc in ADVISOR_MOVES:
                    nr, nc = r + dr, c + dc
                    nsq = rc_to_sq(nr, nc)
                    if nsq not in palace:
                        continue
                    t = self.board[nsq]
                    if t == EMPTY or color_of(t) != color:
                        moves.append(sq * 90 + nsq)
            elif p in (R_KING, B_KING):
                palace = RED_PALACE_SET if p == R_KING else BLACK_PALACE_SET
                for dr, dc in KING_MOVES:
                    nr, nc = r + dr, c + dc
                    nsq = rc_to_sq(nr, nc)
                    if nsq not in palace:
                        continue
                    t = self.board[nsq]
                    if t == EMPTY or color_of(t) != color:
                        moves.append(sq * 90 + nsq)
            elif p in (R_PAWN, B_PAWN):
                # 红兵向上（row-1），黑卒向下（row+1）
                if p == R_PAWN:
                    crossed = r <= 4
                    dirs = [(-1, 0)]
                    if crossed:
                        dirs += [(0, -1), (0, 1)]
                else:
                    crossed = r >= 5
                    dirs = [(1, 0)]
                    if crossed:
                        dirs += [(0, -1), (0, 1)]
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if not in_board(nr, nc):
                        continue
                    nsq = rc_to_sq(nr, nc)
                    t = self.board[nsq]
                    if t == EMPTY or color_of(t) != color:
                        moves.append(sq * 90 + nsq)
        return moves

    def is_attacked(self, sq: int, by_color: int) -> bool:
        """sq 是否被 by_color 攻击（用于将军检测与走法合法性）"""
        r, c = sq_to_rc(sq)
        # 车/炮直线
        for dr, dc in ROOK_DIRS:
            nr, nc = r + dr, c + dc
            while in_board(nr, nc):
                nsq = rc_to_sq(nr, nc)
                p = self.board[nsq]
                if p != EMPTY:
                    if color_of(p) == by_color and p in (R_ROOK, B_ROOK):
                        return True
                    break
                nr += dr
                nc += dc
        # 炮（隔子）
        for dr, dc in ROOK_DIRS:
            nr, nc = r + dr, c + dc
            jumped = False
            while in_board(nr, nc):
                nsq = rc_to_sq(nr, nc)
                p = self.board[nsq]
                if not jumped:
                    if p != EMPTY:
                        jumped = True
                else:
                    if p != EMPTY:
                        if color_of(p) == by_color and p in (R_CANNON, B_CANNON):
                            return True
                        break
                nr += dr
                nc += dc
        # 马
        for (dr, dc), (br, bc) in HORSE_MOVES:
            nr, nc = r + dr, c + dc
            if not in_board(nr, nc):
                continue
            leg = rc_to_sq(r + br, c + bc)
            if self.board[leg] != EMPTY:
                continue
            p = self.board[rc_to_sq(nr, nc)]
            if p != EMPTY and color_of(p) == by_color and p in (R_HORSE, B_HORSE):
                return True
        # 兵/卒
        if by_color == RED:
            # 红兵攻击：向下（row+1）或过河后左右
            for dr, dc in [(1, 0)]:
                nr, nc = r + dr, c + dc
                if in_board(nr, nc):
                    p = self.board[rc_to_sq(nr, nc)]
                    if p == R_PAWN:
                        return True
            if r >= 5:  # 红兵已过河（在对方半场 row<=4 是红兵过河；攻击方视角：红兵在 row<=4 才能横走）
                pass
            # 红兵过河后（row<=4）可横走，攻击左右
            for dr, dc in [(0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if in_board(nr, nc):
                    p = self.board[rc_to_sq(nr, nc)]
                    if p == R_PAWN and nr <= 4:
                        return True
        else:
            for dr, dc in [(-1, 0)]:
                nr, nc = r + dr, c + dc
                if in_board(nr, nc):
                    p = self.board[rc_to_sq(nr, nc)]
                    if p == B_PAWN:
                        return True
            for dr, dc in [(0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if in_board(nr, nc):
                    p = self.board[rc_to_sq(nr, nc)]
                    if p == B_PAWN and nr >= 5:
                        return True
        # 帅/将照面（飞将）：同列且中间无子
        kpos = self._king_pos[by_color]
        if kpos >= 0:
            kr, kc = sq_to_rc(kpos)
            if kc == c:
                lo, hi = min(r, kr), max(r, kr)
                blocked = False
                for rr in range(lo + 1, hi):
                    if self.board[rc_to_sq(rr, c)] != EMPTY:
                        blocked = True
                        break
                if not blocked:
                    return True
        return False

    def in_check(self, color: int) -> bool:
        k = self.find_king(color)
        if k < 0:
            return False
        return self.is_attacked(k, 1 - color)

    def legal_moves(self) -> list:
        """生成全部合法走法（过滤送将）"""
        color = self.turn
        pseudo = self._gen_pseudo_moves(color)
        legal = []
        for mv in pseudo:
            fsq, tsq = divmod(mv, 90)
            captured = self.board[tsq]
            self.board[tsq] = self.board[fsq]
            self.board[fsq] = EMPTY
            # 更新哈希
            self._hash ^= self._zobrist[fsq][self.board[tsq]]
            self._hash ^= self._zobrist[tsq][self.board[tsq]]
            if captured:
                self._hash ^= self._zobrist[tsq][captured]
            if not self.in_check(color):
                legal.append(mv)
            # 回退
            self._hash ^= self._zobrist[fsq][self.board[tsq]]
            self._hash ^= self._zobrist[tsq][self.board[tsq]]
            if captured:
                self._hash ^= self._zobrist[tsq][captured]
            self.board[fsq] = self.board[tsq]
            self.board[tsq] = captured
        return legal

    def make_move(self, mv: int):
        fsq, tsq = divmod(mv, 90)
        captured = self.board[tsq]
        moved = self.board[fsq]
        self._history.append((fsq, tsq, captured, self.halfmove, self.turn, self.fullmove))
        self.board[tsq] = moved
        self.board[fsq] = EMPTY
        self._hash ^= self._zobrist[fsq][moved]
        self._hash ^= self._zobrist[tsq][moved]
        if captured:
            self._hash ^= self._zobrist[tsq][captured]
        self._update_king_pos(fsq, tsq, moved)
        self.halfmove = 0 if captured else self.halfmove + 1
        self.turn = 1 - self.turn
        self._hash ^= self._zobrist_turn
        if self.turn == RED:
            self.fullmove += 1

    def unmake_move(self):
        if not self._history:
            return
        fsq, tsq, captured, halfmove, turn, fullmove = self._history.pop()
        moved = self.board[tsq]
        # 移除 tsq 上的 moved 棋子，添加回 fsq
        self._hash ^= self._zobrist[tsq][moved]
        self._hash ^= self._zobrist[fsq][moved]
        # 被吃子放回 tsq
        if captured:
            self._hash ^= self._zobrist[tsq][captured]
        self.board[fsq] = moved
        self.board[tsq] = captured
        self._update_king_pos(tsq, fsq, moved)
        self.halfmove = halfmove
        self.turn = turn
        self.fullmove = fullmove
        self._hash ^= self._zobrist_turn

    # ---------- 胜负判定 ----------
    def is_game_over(self) -> bool:
        return self.result() is not None

    def result(self):
        """返回 '1-0'（红胜）/ '0-1'（黑胜）/ '1/2-1/2'（和棋）/ None"""
        legal = self.legal_moves()
        if not legal:
            # 无子可走：被将军则输，否则困毙也输
            return '0-1' if self.turn == RED else '1-0'
        # 自然限着：60 回合（120 半回合）无吃子判和
        if self.halfmove >= 120:
            return '1/2-1/2'
        return None

    def move_to_uci(self, mv: int) -> str:
        fsq, tsq = divmod(mv, 90)
        fr, fc = sq_to_rc(fsq)
        tr, tc = sq_to_rc(tsq)
        return f'{chr(ord("a") + fc)}{9 - fr}{chr(ord("a") + tc)}{9 - tr}'

    def uci_to_move(self, uci: str) -> int:
        fc = ord(uci[0]) - ord('a')
        fr = 9 - int(uci[1])
        tc = ord(uci[2]) - ord('a')
        tr = 9 - int(uci[3])
        return rc_to_sq(fr, fc) * 90 + rc_to_sq(tr, tc)

    def move_to_iccs(self, mv: int) -> str:
        """ICCS 坐标（中国象棋标准，a0-i9，红方视角）"""
        fsq, tsq = divmod(mv, 90)
        fr, fc = sq_to_rc(fsq)
        tr, tc = sq_to_rc(tsq)
        return f'{chr(ord("a") + fc)}{fr}{chr(ord("a") + tc)}{tr}'

    def iccs_to_move(self, iccs: str) -> int:
        fc = ord(iccs[0]) - ord('a')
        fr = int(iccs[1])
        tc = ord(iccs[2]) - ord('a')
        tr = int(iccs[3])
        return rc_to_sq(fr, fc) * 90 + rc_to_sq(tr, tc)
