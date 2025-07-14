# action_encoder.py

import random

def encode_action(board, legal_moves):
    """
    将所有合法动作编码为列表。
    """
    return [move.uci() for move in legal_moves]

def decode_action(board, action_index):
    """
    根据索引解码为实际合法动作，加入越界保护。
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")
    if action_index < 0 or action_index >= len(legal_moves):
        # 越界处理：返回第一个合法动作（或随机选一个）
        return legal_moves[0]
    return legal_moves[action_index]

def encode_action_index(board, move):
    """
    返回指定动作在当前合法动作列表中的索引。
    """
    legal_moves = list(board.legal_moves)
    return legal_moves.index(move)


















    
