# -*- coding: utf-8 -*-
"""棋盘绘制组件：深色木质主题，QPainter 绘制"""
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QLinearGradient, QRadialGradient,
    QFont, QPainterPath,
)
from PySide6.QtWidgets import QWidget

from core.board import (
    XiangqiBoard, RED, BLACK, EMPTY, ROWS, COLS,
    PIECE_NAMES, sq_to_rc, rc_to_sq,
)

# 主题色
BG_TOP = QColor(46, 36, 26)        # 深棕背景
BG_BOTTOM = QColor(30, 24, 18)
BOARD_LINE = QColor(212, 175, 120)  # 金色棋盘线
BOARD_LINE_DIM = QColor(160, 130, 90)
RIVER_TEXT = QColor(150, 120, 80)
RED_PIECE = QColor(196, 48, 43)     # 红方棋子
BLACK_PIECE = QColor(40, 40, 40)    # 黑方棋子
PIECE_FACE = QColor(238, 224, 200)  # 棋子底色
PIECE_FACE_EDGE = QColor(200, 180, 150)
HIGHLIGHT = QColor(255, 200, 60, 200)      # 选中高亮
MOVE_DOT = QColor(120, 200, 120, 220)      # 合法走法提示点
LAST_MOVE = QColor(255, 160, 60, 120)      # 上一步标记
CHECK_GLOW = QColor(255, 60, 60, 160)      # 将军提示


class BoardWidget(QWidget):
    """中国象棋棋盘控件"""

    move_made = Signal(int)  # 用户走子信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = XiangqiBoard()
        self.cell = 62          # 格子大小
        self.margin = 40        # 边距
        self.selected = -1      # 选中的格子
        self.legal_targets = [] # 选中棋子的合法目标
        self.last_move = None   # (from_sq, to_sq)
        self.check_sq = -1      # 被将军的将帅位置
        self.human_turn = True  # 是否轮到人类走子
        self.game_over = False
        self.setMinimumSize(self._board_px(), self._board_px())
        self.setMouseTracking(True)

    def _board_px(self) -> int:
        return self.margin * 2 + (COLS - 1) * self.cell

    def _board_px_h(self) -> int:
        return self.margin * 2 + (ROWS - 1) * self.cell

    def _sq_to_point(self, sq: int) -> QPointF:
        r, c = sq_to_rc(sq)
        return QPointF(self.margin + c * self.cell, self.margin + r * self.cell)

    def _point_to_sq(self, x: float, y: float) -> int:
        c = round((x - self.margin) / self.cell)
        r = round((y - self.margin) / self.cell)
        if 0 <= r < ROWS and 0 <= c < COLS:
            return rc_to_sq(r, c)
        return -1

    def set_board(self, board: XiangqiBoard):
        self.board = board
        self.update()

    def refresh(self):
        self.update()

    # ---------- 绘制 ----------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        self._draw_background(p)
        self._draw_board_lines(p)
        self._draw_marks(p)
        self._draw_pieces(p)

    def _draw_background(self, p: QPainter):
        w, h = self.width(), self.height()
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, BG_TOP)
        grad.setColorAt(1, BG_BOTTOM)
        p.fillRect(0, 0, w, h, QBrush(grad))
        # 棋盘底板（略亮的矩形）
        bw = self._board_px()
        bh = self._board_px_h()
        x0 = (w - bw) / 2
        y0 = (h - bh) / 2
        self._board_origin = (x0, y0)
        board_rect = QRectF(x0 - 12, y0 - 12, bw + 24, bh + 24)
        p.setPen(QPen(QColor(90, 70, 50), 2))
        p.setBrush(QBrush(QColor(58, 46, 34)))
        p.drawRoundedRect(board_rect, 10, 10)
        # 内层细边
        p.setPen(QPen(QColor(120, 95, 65), 1))
        p.drawRoundedRect(QRectF(x0 - 6, y0 - 6, bw + 12, bh + 12), 6, 6)

    def _draw_board_lines(self, p: QPainter):
        x0, y0 = self._board_origin
        pen = QPen(BOARD_LINE, 1.6)
        p.setPen(pen)
        # 横线
        for r in range(ROWS):
            y = y0 + r * self.cell
            p.drawLine(QPointF(x0, y), QPointF(x0 + (COLS - 1) * self.cell, y))
        # 竖线（河界处断开）
        for c in range(COLS):
            x = x0 + c * self.cell
            p.drawLine(QPointF(x, y0), QPointF(x, y0 + 4 * self.cell))
            p.drawLine(QPointF(x, y0 + 5 * self.cell), QPointF(x, y0 + 9 * self.cell))
        # 九宫斜线
        for (r1, c1), (r2, c2) in [((0, 3), (2, 5)), ((0, 5), (2, 3)),
                                   ((7, 3), (9, 5)), ((7, 5), (9, 3))]:
            p.drawLine(QPointF(x0 + c1 * self.cell, y0 + r1 * self.cell),
                       QPointF(x0 + c2 * self.cell, y0 + r2 * self.cell))
        # 河界文字
        p.setPen(QPen(RIVER_TEXT, 1))
        font = QFont()
        font.setFamilies(['楷体', 'KaiTi', 'Microsoft YaHei', 'SimSun', ''])
        font.setPointSize(20)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QRectF(x0, y0 + 4 * self.cell, (COLS - 1) * self.cell / 2, self.cell),
                   Qt.AlignCenter, '楚 河')
        p.drawText(QRectF(x0 + (COLS - 1) * self.cell / 2, y0 + 4 * self.cell,
                          (COLS - 1) * self.cell / 2, self.cell),
                   Qt.AlignCenter, '汉 界')
        # 炮位/兵位标记
        p.setPen(QPen(BOARD_LINE_DIM, 1.2))
        for r, c in [(2, 1), (2, 7), (7, 1), (7, 7)]:
            self._draw_corner_mark(p, x0 + c * self.cell, y0 + r * self.cell)
        for r in (3, 6):
            for c in (0, 2, 4, 6, 8):
                self._draw_corner_mark(p, x0 + c * self.cell, y0 + r * self.cell)

    def _draw_corner_mark(self, p: QPainter, x: float, y: float):
        """交叉点的小标记（炮位/兵位）"""
        s = 5
        p.drawLine(QPointF(x - s, y - s), QPointF(x - s * 0.3, y - s))
        p.drawLine(QPointF(x - s, y - s), QPointF(x - s, y - s * 0.3))
        p.drawLine(QPointF(x + s, y - s), QPointF(x + s * 0.3, y - s))
        p.drawLine(QPointF(x + s, y - s), QPointF(x + s, y - s * 0.3))
        p.drawLine(QPointF(x - s, y + s), QPointF(x - s * 0.3, y + s))
        p.drawLine(QPointF(x - s, y + s), QPointF(x - s, y + s * 0.3))
        p.drawLine(QPointF(x + s, y + s), QPointF(x + s * 0.3, y + s))
        p.drawLine(QPointF(x + s, y + s), QPointF(x + s, y + s * 0.3))

    def _draw_marks(self, p: QPainter):
        x0, y0 = self._board_origin
        # 上一步标记
        if self.last_move:
            fsq, tsq = self.last_move
            for sq in (fsq, tsq):
                r, c = sq_to_rc(sq)
                x, y = x0 + c * self.cell, y0 + r * self.cell
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(LAST_MOVE))
                p.drawEllipse(QPointF(x, y), self.cell * 0.42, self.cell * 0.42)
        # 将军提示
        if self.check_sq >= 0:
            r, c = sq_to_rc(self.check_sq)
            x, y = x0 + c * self.cell, y0 + r * self.cell
            p.setPen(QPen(CHECK_GLOW, 3))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(x, y), self.cell * 0.48, self.cell * 0.48)
        # 选中高亮
        if self.selected >= 0:
            r, c = sq_to_rc(self.selected)
            x, y = x0 + c * self.cell, y0 + r * self.cell
            p.setPen(QPen(HIGHLIGHT, 2.5))
            p.setBrush(QBrush(HIGHLIGHT))
            p.drawEllipse(QPointF(x, y), self.cell * 0.45, self.cell * 0.45)
        # 合法走法提示点
        for sq in self.legal_targets:
            r, c = sq_to_rc(sq)
            x, y = x0 + c * self.cell, y0 + r * self.cell
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(MOVE_DOT))
            if self.board.board[sq] == EMPTY:
                p.drawEllipse(QPointF(x, y), self.cell * 0.12, self.cell * 0.12)
            else:
                p.drawEllipse(QPointF(x, y), self.cell * 0.45, self.cell * 0.45)

    def _draw_pieces(self, p: QPainter):
        x0, y0 = self._board_origin
        for sq, piece in enumerate(self.board.board):
            if piece == EMPTY:
                continue
            r, c = sq_to_rc(sq)
            x, y = x0 + c * self.cell, y0 + r * self.cell
            self._draw_piece(p, x, y, piece)

    def _draw_piece(self, p: QPainter, x: float, y: float, piece: int):
        radius = self.cell * 0.44
        # 阴影
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 90)))
        p.drawEllipse(QPointF(x + 2, y + 3), radius, radius)
        # 棋子底（径向渐变）
        grad = QRadialGradient(QPointF(x - radius * 0.3, y - radius * 0.3), radius * 1.6)
        grad.setColorAt(0, PIECE_FACE)
        grad.setColorAt(1, PIECE_FACE_EDGE)
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QColor(120, 100, 75), 1.5))
        p.drawEllipse(QPointF(x, y), radius, radius)
        # 内圈
        p.setPen(QPen(QColor(150, 125, 95), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(x, y), radius * 0.78, radius * 0.78)
        # 文字
        is_red = piece <= 7
        color = RED_PIECE if is_red else BLACK_PIECE
        p.setPen(QPen(color, 1))
        font = QFont()
        font.setFamilies(['楷体', 'KaiTi', 'Microsoft YaHei', 'SimSun', ''])
        font.setPointSize(int(self.cell * 0.42))
        font.setBold(True)
        p.setFont(font)
        name = PIECE_NAMES[piece]
        p.drawText(QRectF(x - radius, y - radius, radius * 2, radius * 2),
                   Qt.AlignCenter, name)

    # ---------- 交互 ----------
    def mousePressEvent(self, event):
        if not self.human_turn or self.game_over:
            return
        x, y = event.position().x(), event.position().y()
        x0, y0 = self._board_origin
        sq = self._point_to_sq(x - x0, y - y0)
        if sq < 0:
            return
        piece = self.board.board[sq]
        if self.selected >= 0:
            # 有选中：尝试走子
            if sq in self.legal_targets:
                self.move_made.emit(sq * 90 + self.selected)  # 注意：move = from*90+to
                return
            # 点击自己棋子：切换选中
            if piece != EMPTY and color_of(piece) == self.board.turn:
                self._select(sq)
                return
            # 点击空白/对方棋子：取消选中
            self._clear_selection()
            return
        # 无选中：选中己方棋子
        if piece != EMPTY and color_of(piece) == self.board.turn:
            self._select(sq)

    def _select(self, sq: int):
        self.selected = sq
        self.legal_targets = []
        for mv in self.board.legal_moves():
            fsq, tsq = divmod(mv, 90)
            if fsq == sq:
                self.legal_targets.append(tsq)
        self.update()

    def _clear_selection(self):
        self.selected = -1
        self.legal_targets = []
        self.update()

    def apply_move(self, mv: int):
        """应用走法并更新界面状态"""
        fsq, tsq = divmod(mv, 90)
        self.board.make_move(mv)
        self.last_move = (fsq, tsq)
        self.selected = -1
        self.legal_targets = []
        # 更新将军提示
        self.check_sq = -1
        if self.board.in_check(self.board.turn):
            self.check_sq = self.board.find_king(self.board.turn)
        self.game_over = self.board.is_game_over()
        self.update()
