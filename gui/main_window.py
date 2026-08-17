# -*- coding: utf-8 -*-
"""主窗口：人机对战、悔棋、重开、提示、难度、棋谱保存与复盘、训练状态"""
import os
import time
import threading

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QListWidgetItem, QGroupBox, QMessageBox,
    QFileDialog, QSplitter, QFrame, QProgressBar,
)

from core.board import XiangqiBoard, RED, BLACK, PIECE_NAMES, sq_to_rc
from gui.board import BoardWidget
from ai.classic_ai import make_classic_ai
from ai.model import XiangqiNet
from ai.mcts import MCTS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

DIFFICULTIES = [
    ('easy', '入门（1层）'),
    ('medium', '中级（2层）'),
    ('hard', '高级（3层）'),
    ('master', '大师（4层）'),
]


class AIWorker(QThread):
    """后台 AI 计算线程"""
    move_ready = Signal(int)
    hint_ready = Signal(int)

    def __init__(self, board: XiangqiBoard, engine, parent=None):
        super().__init__(parent)
        self.board = board
        self.engine = engine
        self._hint = False

    def run(self):
        try:
            mv = self.engine.best_move(self.board)
            if self._hint:
                self.hint_ready.emit(mv)
            else:
                self.move_ready.emit(mv)
        except Exception as e:
            print(f'AI 计算失败: {e}')

    def as_hint(self):
        self._hint = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('中国象棋 · AlphaZero 混合引擎')
        self.resize(1080, 720)

        self.board = XiangqiBoard()
        self.engine_name = 'classic'
        self.difficulty = 'hard'
        self.classic_ai = make_classic_ai(self.difficulty)
        self.mcts = None
        self.ai_worker = None
        self.hint_move = -1
        self.move_list = []          # 本局走法（int 列表）
        self.game_started = False

        self._build_ui()
        self._load_models()
        self._new_game()

    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # 左侧棋盘
        self.board_widget = BoardWidget()
        self.board_widget.move_made.connect(self._on_human_move)
        root.addWidget(self.board_widget, stretch=3)

        # 右侧面板
        panel = QWidget()
        panel.setFixedWidth(320)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(10)

        # 状态
        self.status_label = QLabel()
        self.status_label.setStyleSheet(
            'font-size:16px; font-weight:bold; color:#c8a86a; padding:6px;'
            'background:#2e241a; border-radius:6px;')
        self.status_label.setAlignment(Qt.AlignCenter)
        pl.addWidget(self.status_label)

        # 引擎与难度
        gb = QGroupBox('对局设置')
        gb.setStyleSheet('QGroupBox{color:#c8a86a; font-weight:bold;}')
        gl = QVBoxLayout(gb)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('引擎:'))
        self.engine_combo = QComboBox()
        self.engine_combo.addItem('经典 Alpha-Beta', 'classic')
        self.engine_combo.addItem('神经网络 MCTS', 'mcts')
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        row1.addWidget(self.engine_combo, 1)
        gl.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel('难度:'))
        self.diff_combo = QComboBox()
        for key, name in DIFFICULTIES:
            self.diff_combo.addItem(name, key)
        self.diff_combo.setCurrentIndex(2)
        self.diff_combo.currentIndexChanged.connect(self._on_diff_changed)
        row2.addWidget(self.diff_combo, 1)
        gl.addLayout(row2)
        pl.addWidget(gb)

        # 操作按钮
        btn_style = ('QPushButton{background:#3a2e20; color:#e8d8b0; border:1px solid #6a5438;'
                     'border-radius:6px; padding:8px; font-size:14px;}'
                     'QPushButton:hover{background:#4a3a28;}'
                     'QPushButton:disabled{color:#777;}')
        grid = QHBoxLayout()
        self.btn_new = QPushButton('新对局')
        self.btn_undo = QPushButton('悔棋')
        self.btn_hint = QPushButton('提示')
        for b in (self.btn_new, self.btn_undo, self.btn_hint):
            b.setStyleSheet(btn_style)
        self.btn_new.clicked.connect(self._new_game)
        self.btn_undo.clicked.connect(self._undo)
        self.btn_hint.clicked.connect(self._hint)
        grid.addWidget(self.btn_new)
        grid.addWidget(self.btn_undo)
        grid.addWidget(self.btn_hint)
        pl.addLayout(grid)

        row3 = QHBoxLayout()
        self.btn_save = QPushButton('保存棋谱')
        self.btn_replay = QPushButton('复盘')
        for b in (self.btn_save, self.btn_replay):
            b.setStyleSheet(btn_style)
        self.btn_save.clicked.connect(self._save_pgn)
        self.btn_replay.clicked.connect(self._open_replay)
        row3.addWidget(self.btn_save)
        row3.addWidget(self.btn_replay)
        pl.addLayout(row3)

        # 走法列表
        pl.addWidget(QLabel('走法记录:'))
        self.move_list_widget = QListWidget()
        self.move_list_widget.setStyleSheet(
            'QListWidget{background:#241c14; color:#d8c8a0; border:1px solid #4a3a28;'
            'border-radius:6px; font-size:14px;}')
        pl.addWidget(self.move_list_widget, stretch=2)

        # 训练状态
        gb2 = QGroupBox('训练状态')
        gb2.setStyleSheet('QGroupBox{color:#c8a86a; font-weight:bold;}')
        tl = QVBoxLayout(gb2)
        self.train_label = QLabel('未启动训练')
        self.train_label.setWordWrap(True)
        self.train_label.setStyleSheet('color:#a0a0a0; font-size:12px;')
        tl.addWidget(self.train_label)
        self.btn_train = QPushButton('启动后台训练')
        self.btn_train.setStyleSheet(btn_style)
        self.btn_train.clicked.connect(self._toggle_train)
        tl.addWidget(self.btn_train)
        pl.addWidget(gb2)

        root.addWidget(panel, stretch=1)

        # 菜单
        menubar = self.menuBar()
        m_game = menubar.addMenu('对局')
        m_game.addAction('新对局', self._new_game)
        m_game.addAction('悔棋', self._undo)
        m_game.addAction('走法提示', self._hint)
        m_game.addSeparator()
        m_game.addAction('退出', self.close)
        m_help = menubar.addMenu('帮助')
        m_help.addAction('关于', self._about)

        self._update_status()

    # ---------- 模型加载 ----------
    def _load_models(self):
        """尝试加载神经网络模型（存在则启用 MCTS 引擎）"""
        for name in ('best.pt', 'current.pt'):
            path = os.path.join(MODEL_DIR, name)
            if os.path.exists(path):
                try:
                    net = XiangqiNet(channels=32, blocks=3)
                    from trainer.train import load_model
                    load_model(net, path)
                    self.mcts = MCTS(net, num_simulations=100, dirichlet_eps=0.0)
                    self.train_label.setText(f'已加载模型 {name}，MCTS 引擎可用')
                    return
                except Exception as e:
                    print(f'模型加载失败 {path}: {e}')
        self.train_label.setText('未找到模型，MCTS 引擎不可用（可先训练）')

    # ---------- 对局控制 ----------
    def _new_game(self):
        self._cancel_ai()
        self.board = XiangqiBoard()
        self.move_list = []
        self.hint_move = -1
        self.game_started = True
        self.board_widget.set_board(self.board)
        self.board_widget.human_turn = True
        self.board_widget.game_over = False
        self.board_widget.last_move = None
        self.board_widget.check_sq = -1
        self.move_list_widget.clear()
        self._update_status()

    def _on_human_move(self, mv: int):
        """人类走子（红方）"""
        self._apply_move(mv)
        if not self.board.is_game_over():
            self._start_ai()

    def _apply_move(self, mv: int):
        self.move_list.append(mv)
        self.board_widget.apply_move(mv)
        self._append_move_text(mv)
        self._update_status()

    def _append_move_text(self, mv: int):
        n = len(self.move_list)
        fsq, tsq = divmod(mv, 90)
        fr, fc = sq_to_rc(fsq)
        tr, tc = sq_to_rc(tsq)
        piece = self.board_widget.board.board[tsq] if False else None
        # 用 UCI 风格显示
        text = f'{n}. {chr(ord("a")+fc)}{9-fr}->{chr(ord("a")+tc)}{9-tr}'
        item = QListWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        self.move_list_widget.addItem(item)
        self.move_list_widget.scrollToBottom()

    def _start_ai(self):
        """轮到 AI 走子"""
        self.board_widget.human_turn = False
        self._update_status('AI 思考中…')
        engine = self._make_engine()
        self.ai_worker = AIWorker(self.board, engine)
        self.ai_worker.move_ready.connect(self._on_ai_move)
        self.ai_worker.start()

    def _make_engine(self):
        if self.engine_name == 'mcts' and self.mcts is not None:
            return self.mcts
        return self.classic_ai

    def _on_ai_move(self, mv: int):
        if mv < 0:
            self._update_status('AI 无子可走')
            return
        self._apply_move(mv)
        self.board_widget.human_turn = True
        if self.board.is_game_over():
            self._update_status()
        else:
            self._update_status()

    def _cancel_ai(self):
        if self.ai_worker and self.ai_worker.isRunning():
            self.ai_worker.terminate()
            self.ai_worker.wait(500)
        self.ai_worker = None

    def _undo(self):
        """悔棋：撤销 AI 一步 + 人类一步"""
        if not self.move_list:
            return
        self._cancel_ai()
        # 撤销两步（人类 + AI），若只剩一步则撤销一步
        steps = 2 if len(self.move_list) >= 2 else 1
        for _ in range(steps):
            if not self.move_list:
                break
            self.move_list.pop()
            self.board.unmake_move()
            if self.move_list_widget.count() > 0:
                self.move_list_widget.takeItem(self.move_list_widget.count() - 1)
        self.board_widget.set_board(self.board)
        self.board_widget.last_move = None
        self.board_widget.check_sq = -1
        self.board_widget.human_turn = True
        self.board_widget.game_over = False
        self.hint_move = -1
        self._update_status()

    def _hint(self):
        """走法提示：后台计算最佳走法并高亮"""
        if self.board.is_game_over() or not self.move_list:
            return
        if self.ai_worker and self.ai_worker.isRunning():
            return
        self._update_status('计算提示中…')
        engine = self._make_engine()
        self.ai_worker = AIWorker(self.board, engine)
        self.ai_worker.as_hint()
        self.ai_worker.hint_ready.connect(self._on_hint_ready)
        self.ai_worker.start()

    def _on_hint_ready(self, mv: int):
        if mv < 0:
            self._update_status()
            return
        self.hint_move = mv
        fsq, tsq = divmod(mv, 90)
        self.board_widget.selected = fsq
        self.board_widget.legal_targets = [tsq]
        self.board_widget.update()
        self._update_status('提示：高亮为推荐走法')

    # ---------- 状态 ----------
    def _update_status(self, extra: str = None):
        if extra:
            self.status_label.setText(extra)
            return
        if self.board.is_game_over():
            result = self.board.result()
            if result == '1-0':
                self.status_label.setText('红方胜！')
            elif result == '0-1':
                self.status_label.setText('黑方胜！')
            else:
                self.status_label.setText('和棋')
            self.board_widget.human_turn = False
            return
        turn = '红方' if self.board.turn == RED else '黑方'
        check = '（将军！）' if self.board.in_check(self.board.turn) else ''
        self.status_label.setText(f'{turn}走子{check}')

    # ---------- 设置变更 ----------
    def _on_engine_changed(self, idx):
        self.engine_name = self.engine_combo.itemData(idx)
        if self.engine_name == 'mcts' and self.mcts is None:
            QMessageBox.information(self, '提示', '未找到训练模型，已切换回经典引擎。')
            self.engine_combo.setCurrentIndex(0)
            return
        self._new_game()

    def _on_diff_changed(self, idx):
        self.difficulty = self.diff_combo.itemData(idx)
        self.classic_ai = make_classic_ai(self.difficulty)

    # ---------- 棋谱 ----------
    def _save_pgn(self):
        if not self.move_list:
            QMessageBox.information(self, '提示', '本局还没有走法。')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '保存棋谱', os.path.join(PROJECT_ROOT, 'games', 'game.pgn'),
            'PGN 文件 (*.pgn)')
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = [
            '[Event "Casual Game"]',
            '[Site "Marvis Xiangqi"]',
            '[Date "%s"]' % time.strftime('%Y.%m.%d'),
            '[Result "%s"]' % self.board.result(),
            '',
        ]
        for i, mv in enumerate(self.move_list):
            fsq, tsq = divmod(mv, 90)
            fr, fc = sq_to_rc(fsq)
            tr, tc = sq_to_rc(tsq)
            uci = f'{chr(ord("a")+fc)}{9-fr}{chr(ord("a")+tc)}{9-tr}'
            lines.append(uci)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        QMessageBox.information(self, '已保存', f'棋谱已保存到:\n{path}')

    def _open_replay(self):
        if not self.move_list:
            QMessageBox.information(self, '提示', '本局还没有走法可复盘。')
            return
        from gui.replay import ReplayDialog
        dlg = ReplayDialog(self.move_list, self)
        dlg.exec()

    # ---------- 训练 ----------
    def _toggle_train(self):
        if hasattr(self, '_trainer') and self._trainer is not None:
            self._stop_train()
            return
        self._start_train()

    def _start_train(self):
        try:
            from trainer.train_worker import Trainer
            self._trainer = Trainer(
                model_dir=MODEL_DIR, log_dir=LOG_DIR,
                num_workers=2, num_simulations=50,
            )
            self._trainer.start_workers()
            self.btn_train.setText('停止训练')
            self.train_label.setText('后台训练运行中（2 进程）…')
            self._train_timer = QTimer(self)
            self._train_timer.timeout.connect(self._refresh_train_status)
            self._train_timer.start(3000)
        except Exception as e:
            QMessageBox.warning(self, '训练启动失败', str(e))

    def _refresh_train_status(self):
        if not hasattr(self, '_trainer') or self._trainer is None:
            return
        t = self._trainer
        self.train_label.setText(
            f'训练中: 对局 {t.total_games} | 样本 {t.total_samples} | 步数 {t.train_steps}')

    def _stop_train(self):
        if hasattr(self, '_trainer') and self._trainer is not None:
            self._trainer.stop()
            self._trainer = None
            if hasattr(self, '_train_timer'):
                self._train_timer.stop()
        self.btn_train.setText('启动后台训练')
        self.train_label.setText('训练已停止')

    def _about(self):
        QMessageBox.about(
            self, '关于',
            '中国象棋 · AlphaZero 混合引擎\n\n'
            '经典 Alpha-Beta 基线 + AlphaZero 式自我对弈强化学习。\n'
            '纯 CPU 环境，PySide6 桌面应用。')

    def closeEvent(self, event):
        self._cancel_ai()
        self._stop_train()
        super().closeEvent(event)
