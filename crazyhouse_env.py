import chess.variant
import numpy as np
from action_encoder import encode_action, decode_action

class CrazyHouseEnv:
    def __init__(self):
        self.board = chess.variant.CrazyhouseBoard()

    def reset(self):
        self.board.reset()
        return self._get_observation()

    def step(self, action_index):
        move = decode_action(action_index, self.board)

        if move not in self.board.legal_moves:
            return self._get_observation(), -1.0, True, {}

        self.board.push(move)
        reward = 1.0 if self.board.is_checkmate() else 0.0
        done = self.board.is_game_over()
        return self._get_observation(), reward, done, {}

    def legal_actions(self):
        return [encode_action(move, self.board) for move in self.board.legal_moves]

    def _get_observation(self):
        return np.zeros((8, 8), dtype=np.float32)






