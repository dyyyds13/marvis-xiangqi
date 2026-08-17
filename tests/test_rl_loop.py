# -*- coding: utf-8 -*-
"""M2 验证：网络 + MCTS + 自我对弈 + 训练闭环（单进程小规模冒烟）"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ai.model import XiangqiNet, count_parameters
from ai.mcts import MCTS
from trainer.self_play import self_play_game
from trainer.train import ReplayBuffer, train_loop

# 1. 网络初始化
net = XiangqiNet(channels=32, blocks=3)
print(f'网络参数量: {count_parameters(net):,}')

# 2. 单次推理性能
import random
from core.board import XiangqiBoard
b = XiangqiBoard()
t0 = time.time()
for _ in range(20):
    p, v = net.predict(b)
dt = time.time() - t0
print(f'单次推理: {dt/20*1000:.1f}ms')

# 3. MCTS 搜索
mcts = MCTS(net, num_simulations=50, dirichlet_eps=0.25)
t0 = time.time()
probs = mcts.search(b, temperature=1.0)
dt = time.time() - t0
print(f'MCTS 50 次模拟: {dt:.1f}s, 走法数: {len(probs)}')

# 4. 自我对弈一局
t0 = time.time()
samples = self_play_game(net, mcts, max_moves=100)
dt = time.time() - t0
print(f'自我对弈一局: {dt:.1f}s, 样本数: {len(samples)}')

# 5. 训练几步
buffer = ReplayBuffer()
buffer.add(samples)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
t0 = time.time()
history = train_loop(net, buffer, optimizer, num_steps=20, batch_size=32, log_every=10)
dt = time.time() - t0
print(f'训练 20 步: {dt:.1f}s, 最终 loss: {history[-1]["loss"]:.4f}' if history else '样本不足')

print('\nM2 闭环验证完成')
