import numpy as np
import chess
import chess.variant
from action_encoder import encode_action, decode_action

class CrazyHouseEnv:
    def __init__(self):
        self.board = chess.variant.CrazyhouseBoard()
        self.previous_piece_count = len(self.board.piece_map())

    def reset(self):
        self.board.reset()
        self.previous_piece_count = len(self.board.piece_map())
        return self._get_observation()

    def _get_observation(self):
        # 返回 (8, 8) 数组，简单编码：空格 0，白方正数，黑方负数
        obs = np.zeros((8, 8), dtype=np.float32)
        for square, piece in self.board.piece_map().items():
            row, col = divmod(square, 8)
            value = piece.piece_type
            obs[row][col] = value if piece.color == chess.WHITE else -value
        return obs

    def step(self, action_index):
        move = decode_action(action_index, self.board)
        info = {}

        if move not in self.board.legal_moves:
            # 非法动作，直接惩罚
            reward = -1.0
            done = True
            return self._get_observation(), reward, done, info

        prev_piece_count = len(self.board.piece_map())
        self.board.push(move)

        done = self.board.is_game_over()
        result = self.board.result() if done else None

        # 奖励机制
        reward = 0.0

        # 吃子检测
        curr_piece_count = len(self.board.piece_map())
        if curr_piece_count < prev_piece_count:
            reward += 0.2

        # 基础移动奖励
        reward += 0.05

        # 结束奖励
        if done:
            if result == '1-0' and self.board.turn == chess.BLACK:
                reward = 1.0  # 白赢
            elif result == '0-1' and self.board.turn == chess.WHITE:
                reward = 1.0  # 黑赢
            else:
                reward = -1.0  # 平局/自己输

        return self._get_observation(), reward, done, info

    def legal_actions(self):
        legal = [encode_action(move, self.board) for move in self.board.legal_moves]
        return legal

    def decode_action(self, index):
        return decode_action(index, self.board)

