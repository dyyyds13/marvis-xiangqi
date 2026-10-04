
# -*- coding: utf-8 -*-
"""中国象棋 · AlphaZero 混合引擎 —— 程序入口"""
import sys
import os
sys.stdout.reconfigure(line_buffering=True) # 强制行缓冲
print("=" * 50)
print("Python 解释器:", sys.executable)
print("Python 版本:", sys.version)
print("环境前缀:", sys.prefix)
print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV"))
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
print("=" * 50)
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
