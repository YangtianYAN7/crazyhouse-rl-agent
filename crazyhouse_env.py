import chess
import numpy as np
from action_encoder import encode_action, decode_action

class CrazyhouseEnv:
    def __init__(self):
        self.board = chess.Board()

        # 1. 初始化 action_space 和 observation_space
        self.action_space = type('', (), {})()  # 创建空对象
        self.action_space.n = len(encode_action(self.board, self.board.legal_moves))  # 动作数量

        self.observation_space = type('', (), {})()
        self.observation_space.shape = (6, 8, 8)  # 6通道（不同棋子类型）+ 8x8棋盘

    def reset(self):
        self.board.reset()
        return self.get_observation()

    def step(self, action_index):
        move = decode_action(self.board, action_index)

        if move not in self.board.legal_moves:
            return self.get_observation(), -1.0, True, {}

        self.board.push(move)

        reward = 0.0
        done = False

        if self.board.is_checkmate():
            reward = 1.0
            done = True
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = 0.0
            done = True
        elif self.board.is_game_over():
            reward = -1.0
            done = True
        else:
            reward = 0.01  # 小奖励鼓励探索

        return self.get_observation(), reward, done, {}

    def get_observation(self):
        piece_map = self.board.piece_map()
        obs = np.zeros((6, 8, 8), dtype=np.float32)

        piece_to_index = {
            'P': 0, 'N': 1, 'B': 2,
            'R': 3, 'Q': 4, 'K': 5,
        }

        for square, piece in piece_map.items():
            row = 7 - (square // 8)
            col = square % 8
            idx = piece_to_index[piece.symbol().upper()]
            obs[idx, row, col] = 1 if piece.color == chess.WHITE else -1

        return obs

























