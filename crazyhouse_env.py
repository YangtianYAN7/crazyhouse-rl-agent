# crazyhouse_env.py - 支持真正的 Crazyhouse 落子规则

import chess
import numpy as np
from chess.variant import CrazyhouseBoard
from action_encoder import encode_action, decode_action, ALL_POSSIBLE_MOVES

class CrazyhouseEnv:
    def __init__(self):
        self.board = CrazyhouseBoard()
        self.update_action_space()
        self.observation_space = type('', (), {})()
        self.observation_space.shape = (11, 8, 8)
        self.step_count = 0

    def update_action_space(self):
        self.action_space = type('', (), {})()
        self.action_space.n = len(ALL_POSSIBLE_MOVES)

    def reset(self):
        self.board = CrazyhouseBoard()
        self.step_count = 0
        self.update_action_space()
        return self.get_observation()

    def step(self, action_index):
        move = decode_action(self.board, action_index)

        if move not in self.board.legal_moves:
            return self.get_observation(), -1.0, True, {}

        prev_material = self.count_material()
        self.board.push(move)
        self.step_count += 1
        self.update_action_space()

        reward = 0.0
        done = False

        if self.step_count >= 50:
            return self.get_observation(), -1.0, True, {}

        if self.board.is_checkmate():
            return self.get_observation(), 1.0, True, {}
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return self.get_observation(), 0.0, True, {}

        reward += 0.005

        new_material = self.count_material()
        if new_material < prev_material:
            reward += 0.5

        center = [chess.D4, chess.E4, chess.D5, chess.E5]
        reward += 0.05 * sum([self.board.is_attacked_by(self.board.turn, sq) for sq in center])

        if self.board.is_check():
            reward += 0.3

        return self.get_observation(), reward, done, {}

    def count_material(self):
        return sum(piece.piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                        chess.ROOK, chess.QUEEN]
                   for piece in self.board.piece_map().values()
                   if piece.color != self.board.turn)

    def get_observation(self):
        obs = np.zeros((11, 8, 8), dtype=np.float32)
        piece_map = self.board.piece_map()
        piece_to_idx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

        for square, piece in piece_map.items():
            row = 7 - (square // 8)
            col = square % 8
            idx = piece_to_idx[piece.symbol().upper()]
            obs[idx, row, col] = 1 if piece.color == chess.WHITE else -1

        # pockets（库存）作为额外通道，按 P/N/B/R/Q 顺序统计
        for i, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]):
            w = self.board.pockets[chess.WHITE].count(pt)
            b = self.board.pockets[chess.BLACK].count(pt)
            v = w - b if self.board.turn == chess.WHITE else b - w
            obs[6 + i, :, :] = v / 5.0

        return obs

    def get_legal_action_indices(self):
        return [ALL_POSSIBLE_MOVES.index(move.uci()) for move in self.board.legal_moves]


















































