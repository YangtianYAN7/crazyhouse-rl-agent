import chess
import chess.variant
import numpy as np
from action_encoder import encode_action, decode_action

class CrazyHouseEnv:
    def __init__(self):
        self.board = chess.variant.CrazyhouseBoard()

    def reset(self):
        self.board.reset()
        return self.get_obs()

    def get_obs(self):
        obs = np.zeros((8, 8), dtype=np.int8)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                obs[chess.square_rank(square)][chess.square_file(square)] = piece.piece_type * (1 if piece.color == chess.WHITE else -1)
        return obs

    def step(self, action_index):
        move = decode_action(action_index)
        if move not in self.board.legal_moves:
            # 非法动作惩罚
            return self.get_obs(), -1.0, True, {}

        self.board.push(move)

        reward = 0.0
        done = False

        if self.board.is_game_over():
            done = True
            result = self.board.result()
            if result == "1-0":
                reward = 1.0 if self.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                reward = 1.0 if self.board.turn == chess.WHITE else -1.0
            else:
                reward = 0.0

        return self.get_obs(), reward, done, {}

    def legal_actions(self):
        return [encode_action(move) for move in self.board.legal_moves]




