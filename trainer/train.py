# -*- coding: utf-8 -*-
"""训练循环：从回放池采样，优化策略交叉熵 + 价值 MSE"""
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai.model import XiangqiNet, board_to_tensor, MOVE_DIM


class ReplayBuffer:
    """经验回放池（内存版）"""

    def __init__(self, capacity: int = 200000):
        self.capacity = capacity
        self.buffer = []

    def add(self, samples: list):
        self.buffer.extend(samples)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


def train_step(net: XiangqiNet, batch, optimizer, device='cpu') -> dict:
    """训练一步，返回损失"""
    net.train()
    tensors = torch.stack([s[0] for s in batch]).to(device)
    # 策略目标：构建 8100 维 one-hot（按概率分布）
    policy_targets = torch.zeros(len(batch), MOVE_DIM, device=device)
    for i, (_, probs, _) in enumerate(batch):
        for mv, p in probs.items():
            policy_targets[i, mv] = p
    value_targets = torch.tensor([s[2] for s in batch], dtype=torch.float32, device=device)

    policy_logits, value = net(tensors)
    # 策略损失：交叉熵（对合法走法掩码后的分布）
    policy_loss = -torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1), dim=1).mean()
    value_loss = F.mse_loss(value.squeeze(1), value_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        'loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
    }


def train_loop(net: XiangqiNet, buffer: ReplayBuffer, optimizer, num_steps: int,
               batch_size: int = 256, device='cpu', log_every: int = 50,
               log_callback=None) -> list:
    """训练 num_steps 步，返回损失历史"""
    history = []
    for step in range(num_steps):
        if len(buffer) < batch_size:
            break
        batch = buffer.sample(batch_size)
        stats = train_step(net, batch, optimizer, device)
        history.append(stats)
        if log_every and (step + 1) % log_every == 0:
            msg = (f'step {step+1}: loss={stats["loss"]:.4f} '
                   f'ploss={stats["policy_loss"]:.4f} vloss={stats["value_loss"]:.4f}')
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
    return history


def save_model(net: XiangqiNet, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 原子写入：先写临时文件再 rename，避免推理服务进程读到半写文件
    tmp = path + '.tmp'
    try:
        torch.save(net.state_dict(), tmp)
        os.replace(tmp, path)
    except Exception as e:
        # 中断/并发场景下 tmp 可能不完整或已被消费（如 Ctrl+C 打断保存，
        # finally 再存时 tmp 已被 rename），保存失败不应让训练崩溃，
        # 打印警告继续（下次保存会重写）。
        print(f'[save_model] 保存 {path} 失败（跳过，训练继续）: '
              f'{type(e).__name__}: {e}', flush=True)


def load_model(net: XiangqiNet, path: str, device='cpu'):
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net
