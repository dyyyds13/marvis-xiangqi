# -*- coding: utf-8 -*-
"""神经网络：策略+价值双头小型 CNN（CPU 友好）"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.board import (
    XiangqiBoard, RED, BLACK, EMPTY, SQUARES, COLS, ROWS,
    R_KING, R_ADVISOR, R_ELEPHANT, R_HORSE, R_ROOK, R_CANNON, R_PAWN,
    B_KING, B_ADVISOR, B_ELEPHANT, B_HORSE, B_ROOK, B_CANNON, B_PAWN,
    rc_to_sq, sq_to_rc,
)

# 通道数：7 红 + 7 黑 + 1 轮到方 + 1 将军
NUM_CHANNELS = 16
# 走法空间：from_sq * 90 + to_sq
MOVE_DIM = SQUARES * SQUARES  # 8100

# 红方棋子 -> 通道 0-6
RED_CHANNEL = {
    R_KING: 0, R_ADVISOR: 1, R_ELEPHANT: 2, R_HORSE: 3,
    R_ROOK: 4, R_CANNON: 5, R_PAWN: 6,
}
# 黑方棋子 -> 通道 7-13
BLACK_CHANNEL = {
    B_KING: 7, B_ADVISOR: 8, B_ELEPHANT: 9, B_HORSE: 10,
    B_ROOK: 11, B_CANNON: 12, B_PAWN: 13,
}


def board_to_tensor(board: XiangqiBoard) -> torch.Tensor:
    """棋盘 -> 张量 [16, 10, 9]（row, col 维度）"""
    t = torch.zeros(NUM_CHANNELS, ROWS, COLS, dtype=torch.float32)
    for sq, p in enumerate(board.board):
        if p == EMPTY:
            continue
        r, c = sq_to_rc(sq)
        if p <= 7:
            t[RED_CHANNEL[p], r, c] = 1.0
        else:
            t[BLACK_CHANNEL[p], r, c] = 1.0
    t[14] = 1.0 if board.turn == RED else 0.0
    if board.in_check(board.turn):
        t[15] = 1.0
    return t


def legal_move_mask(board: XiangqiBoard) -> torch.Tensor:
    """合法走法掩码 [8100]"""
    mask = torch.zeros(MOVE_DIM, dtype=torch.float32)
    for mv in board.legal_moves():
        mask[mv] = 1.0
    return mask


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class XiangqiNet(nn.Module):
    """策略+价值双头网络"""

    def __init__(self, channels: int = 32, blocks: int = 3):
        super().__init__()
        self.conv_in = nn.Conv2d(NUM_CHANNELS, channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        # 策略头（低秩分解，避免 1440->8100 全连接参数爆炸）
        self.policy_conv = nn.Conv2d(channels, 16, 1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc1 = nn.Linear(16 * ROWS * COLS, 128)
        self.policy_fc2 = nn.Linear(128, MOVE_DIM)

        # 价值头
        self.value_conv = nn.Conv2d(channels, 8, 1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.blocks:
            x = block(x)

        # 策略
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        p = F.relu(self.policy_fc1(p))
        p = self.policy_fc2(p)  # [B, 8100]

        # 价值
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # [-1, 1]

        return p, v

    def predict(self, board: XiangqiBoard):
        """单局面推理，返回 (策略概率[8100], 价值)"""
        self.eval()
        with torch.no_grad():
            t = board_to_tensor(board).unsqueeze(0)
            p, v = self.forward(t)
            p = F.softmax(p, dim=1).squeeze(0)
            return p, v.item()

    def predict_batch(self, boards):
        """批量推理，返回 (策略概率 [N,8100], 价值 [N])"""
        self.eval()
        with torch.no_grad():
            ts = torch.stack([board_to_tensor(b) for b in boards])
            p, v = self.forward(ts)
            p = F.softmax(p, dim=1)
            return p, v.squeeze(1)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
