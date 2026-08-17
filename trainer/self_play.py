# -*- coding: utf-8 -*-
"""自我对弈：生成训练样本"""
import random
from core.board import XiangqiBoard, RED, BLACK
from ai.mcts import MCTS
from ai.model import XiangqiNet, board_to_tensor


def self_play_game(net: XiangqiNet, mcts: MCTS, max_moves: int = 200,
                   temperature_schedule=None) -> list:
    """单局自我对弈，返回样本列表 [(board_tensor, policy, z)]"""
    if temperature_schedule is None:
        temperature_schedule = lambda ply: 1.0 if ply < 30 else (0.5 if ply < 60 else 0.1)
    board = XiangqiBoard()
    samples = []  # (tensor, policy_dict, current_player)
    ply = 0
    while not board.is_game_over() and ply < max_moves:
        temp = temperature_schedule(ply)
        probs = mcts.search(board, temperature=temp)
        # 按概率采样
        moves = list(probs.keys())
        weights = [probs[m] for m in moves]
        mv = random.choices(moves, weights=weights, k=1)[0]
        # 记录样本（当前方视角）
        samples.append((board_to_tensor(board), probs, board.turn))
        board.make_move(mv)
        ply += 1
    # 确定结果
    result = board.result()
    if result == '1-0':
        outcome = 1.0  # 红胜
    elif result == '0-1':
        outcome = -1.0
    else:
        outcome = 0.0
    # 为每个样本计算 z（当前方视角）
    labeled = []
    for tensor, probs, player in samples:
        z = outcome if player == RED else -outcome
        labeled.append((tensor, probs, z))
    return labeled


def self_play_batch(net: XiangqiNet, mcts: MCTS, num_games: int = 1,
                    max_moves: int = 200) -> list:
    """批量自我对弈，返回合并样本"""
    all_samples = []
    for _ in range(num_games):
        all_samples.extend(self_play_game(net, mcts, max_moves))
    return all_samples
