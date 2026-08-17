# -*- coding: utf-8 -*-
"""棋谱复盘窗口：逐步回放、自动播放、点击跳转"""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QSlider, QMessageBox,
)

from core.board import XiangqiBoard, RED, BLACK, sq_to_rc
from gui.board import BoardWidget


class ReplayDialog(QDialog):
    def __init__(self, move_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle('棋谱复盘')
        self.resize(900, 640)
        self.move_list = list(move_list)
        self.step = 0          # 已应用的走法数
        self.board = XiangqiBoard()
        self.playing = False

        root = QVBoxLayout(self)
        top = QHBoxLayout()

        # 棋盘
        self.board_widget = BoardWidget()
        self.board_widget.human_turn = False
        self.board_widget.set_board(self.board)
        top.addWidget(self.board_widget, stretch=3)

        # 右侧走法列表
        right = QVBoxLayout()
        right.addWidget(QLabel('走法序列:'))
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        right.addWidget(self.list_widget, stretch=1)
        top.addLayout(right, stretch=1)
        root.addLayout(top)

        # 控制条
        ctrl = QHBoxLayout()
        self.btn_prev = QPushButton('上一步')
        self.btn_play = QPushButton('自动播放')
        self.btn_next = QPushButton('下一步')
        self.btn_reset = QPushButton('回到开头')
        for b in (self.btn_prev, self.btn_play, self.btn_next, self.btn_reset):
            b.setStyleSheet(
                'QPushButton{background:#3a2e20; color:#e8d8b0; border:1px solid #6a5438;'
                'border-radius:6px; padding:6px 12px;}'
                'QPushButton:hover{background:#4a3a28;}')
        self.btn_prev.clicked.connect(self._prev)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_next.clicked.connect(self._next)
        self.btn_reset.clicked.connect(self._reset)
        ctrl.addWidget(self.btn_reset)
        ctrl.addWidget(self.btn_prev)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_next)
        root.addLayout(ctrl)

        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(self.move_list))
        self.slider.valueChanged.connect(self._on_slider)
        root.addWidget(self.slider)

        self.step_label = QLabel()
        self.step_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.step_label)

        # 填充走法列表
        for i, mv in enumerate(self.move_list):
            fsq, tsq = divmod(mv, 90)
            fr, fc = sq_to_rc(fsq)
            tr, tc = sq_to_rc(tsq)
            text = f'{i+1}. {chr(ord("a")+fc)}{9-fr}->{chr(ord("a")+tc)}{9-tr}'
            item = QListWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            self.list_widget.addItem(item)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next)
        self._timer.setInterval(800)

        self._refresh()

    # ---------- 控制 ----------
    def _reset(self):
        self._stop_play()
        self.step = 0
        self.board = XiangqiBoard()
        self._refresh()

    def _prev(self):
        self._stop_play()
        if self.step > 0:
            self.step -= 1
            self._rebuild()
            self._refresh()

    def _next(self):
        if self.step >= len(self.move_list):
            self._stop_play()
            return
        self.step += 1
        self._rebuild()
        self._refresh()
        if self.step >= len(self.move_list):
            self._stop_play()

    def _toggle_play(self):
        if self.playing:
            self._stop_play()
        else:
            if self.step >= len(self.move_list):
                self._reset()
            self.playing = True
            self.btn_play.setText('暂停')
            self._timer.start()

    def _stop_play(self):
        self.playing = False
        self._timer.stop()
        self.btn_play.setText('自动播放')

    def _rebuild(self):
        """从走法序列重建到第 step 步的局面"""
        self.board = XiangqiBoard()
        for mv in self.move_list[:self.step]:
            self.board.make_move(mv)

    def _on_row_changed(self, row):
        if row < 0:
            return
        self._stop_play()
        self.step = row + 1
        self._rebuild()
        self._refresh()

    def _on_slider(self, val):
        if val == self.step:
            return
        self._stop_play()
        self.step = val
        self._rebuild()
        self._refresh()

    def _refresh(self):
        self.board_widget.set_board(self.board)
        # 上一步标记
        if self.step > 0:
            mv = self.move_list[self.step - 1]
            fsq, tsq = divmod(mv, 90)
            self.board_widget.last_move = (fsq, tsq)
        else:
            self.board_widget.last_move = None
        # 将军提示
        self.board_widget.check_sq = -1
        if self.board.in_check(self.board.turn):
            self.board_widget.check_sq = self.board.find_king(self.board.turn)
        self.board_widget.update()
        self.slider.blockSignals(True)
        self.slider.setValue(self.step)
        self.slider.blockSignals(False)
        if self.step > 0:
            self.list_widget.setCurrentRow(self.step - 1)
        total = len(self.move_list)
        self.step_label.setText(f'第 {self.step} / {total} 步')
