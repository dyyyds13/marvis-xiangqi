# -*- coding: utf-8 -*-
"""评估：新模型 vs 基线/旧模型，胜率达标才替换"""
import random
from core.board import XiangqiBoard, RED, BLACK
from ai.mcts import MCTS
from ai.classic_ai import ClassicAI


def play_game(ai_red, ai_black, max_moves: int = 200, mcts_sims: int = 100) -> str:
    """AI 对弈一局，返回结果 '1-0'/'0-1'/'1/2-1/2'"""
    board = XiangqiBoard()
    ply = 0
    while not board.is_game_over() and ply < max_moves:
        if board.turn == RED:
            mv = ai_red(board)
        else:
            mv = ai_black(board)
        if mv is None or mv < 0:
            break
        board.make_move(mv)
        ply += 1
    return board.result() or '1/2-1/2'


def make_mcts_player(net, num_simulations: int = 100):
    mcts = MCTS(net, num_simulations=num_simulations, dirichlet_eps=0.0)
    def player(board):
        return mcts.best_move(board, temperature=0.0)
    return player


def make_classic_player(depth: int = 3, time_limit: float = 1.0):
    ai = ClassicAI(depth=depth, time_limit=time_limit)
    def player(board):
        return ai.best_move(board)
    return player


def evaluate_net(net, baseline_player, num_games: int = 20, num_simulations: int = 100,
                 as_red: bool = True) -> dict:
    """评估网络 vs 基线，返回胜率统计（网络视角）"""
    mcts_player = make_mcts_player(net, num_simulations)
    wins = draws = losses = 0
    for _ in range(num_games):
        if as_red:
            result = play_game(mcts_player, baseline_player)
        else:
            result = play_game(baseline_player, mcts_player)
        if result == '1-0':
            if as_red:
                wins += 1
            else:
                losses += 1
        elif result == '0-1':
            if as_red:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1
    total = wins + draws + losses
    return {
        'wins': wins, 'draws': draws, 'losses': losses,
        'win_rate': wins / total if total else 0.0,
        'score': (wins + 0.5 * draws) / total if total else 0.0,
    }
