# -*- coding: utf-8 -*-
"""经典 AI 自战测试"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.board import XiangqiBoard
from ai.classic_ai import ClassicAI

ai = ClassicAI(depth=2, time_limit=0.5)
b = XiangqiBoard()
t0 = time.time()
for i in range(60):
    if b.is_game_over():
        break
    mv = ai.best_move(b)
    if mv < 0:
        break
    b.make_move(mv)
dt = time.time() - t0
print(f'经典 AI 自战 {i+1} 步, 耗时 {dt:.1f}s, 结果: {b.result()}')
print(f'平均 {dt/(i+1)*1000:.0f}ms/步')
