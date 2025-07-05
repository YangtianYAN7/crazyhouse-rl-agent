import numpy as np
import chess.variant
from action_encoder import encode_action, decode_action, NUM_ACTIONS

class CrazyHouseEnv:
    def __init__(self):
        self.board = chess.variant.CrazyhouseBoard()

    def reset(self):
        self.board.reset()
        return self._get_observation()

    def step(self, action_index):
        move = decode_action(action_index, self.board)
        if move is None or not self.board.is_legal(move):
            return self._get_observation(), -1.0, True, {}

        self.board.push(move)

        reward = 0.0
        done = self.board.is_game_over()
        if done:
            outcome = self.board.outcome()
            if outcome is not None:
                if outcome.winner is True:
                    reward = 1.0
                elif outcome.winner is False:
                    reward = -1.0

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # One-hot encode piece planes, shape: (12, 8, 8)
        obs = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                plane = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                obs[plane, rank, file] = 1.0
        return obs

    @property
    def legal_actions(self):
        return [encode_action(move, self.board) for move in self.board.legal_moves if encode_action(move, self.board) >= 0]








