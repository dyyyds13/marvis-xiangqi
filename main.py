# -*- coding: utf-8 -*-
"""中国象棋 · AlphaZero 混合引擎 —— 程序入口"""
import sys
import os

# 确保项目根目录在 sys.path（支持从任意目录启动）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('中国象棋')
    app.setOrganizationName('Marvis')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
