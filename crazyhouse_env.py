# crazyhouse_env.py - 增加阶段性训练支持（控制是否允许 drop）

import chess
import numpy as np
from chess.variant import CrazyhouseBoard
from action_encoder import ALL_POSSIBLE_MOVES
from action_encoder import decode_action

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

class CrazyhouseEnv:
    def __init__(self, allow_drops=True):
        self.board = CrazyhouseBoard()
        self.allow_drops = allow_drops
        self.update_action_space()
        self.observation_space = type('', (), {})()
        self.observation_space.shape = (11, 8, 8)
        self.step_count = 0
        self.last_check = False

    def update_action_space(self):
        self.action_space = type('', (), {})()
        self.action_space.n = len(ALL_POSSIBLE_MOVES)

    def reset(self):
        self.board = CrazyhouseBoard()
        self.step_count = 0
        self.last_check = False
        self.update_action_space()
        return self.get_observation()

    def step(self, action_index):
        move = decode_action(self.board, action_index)

        if move not in self.board.legal_moves:
            return self.get_observation(), -1.0, True, {}

        prev_material = self.count_material()
        prev_pockets = self.count_pockets()
        prev_step = self.step_count

        self.board.push(move)
        self.step_count += 1
        self.update_action_space()

        reward = 0.0
        done = False

        if self.step_count >= 60:
            return self.get_observation(), -1.0, True, {}

        if self.board.is_checkmate():
            reward += 5.0 + (60 - self.step_count) * 0.1
            return self.get_observation(), reward, True, {}
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return self.get_observation(), -0.2, True, {}

        reward += 0.005

        new_material = self.count_material()
        if new_material < prev_material:
            reward += abs(prev_material - new_material) * 0.1

        new_pockets = self.count_pockets()
        if new_pockets < prev_pockets:
            reward += (prev_pockets - new_pockets) * 0.2

        center = [chess.D4, chess.E4, chess.D5, chess.E5]
        reward += 0.05 * sum([self.board.is_attacked_by(self.board.turn, sq) for sq in center])

        opp_king_square = self.get_opponent_king_square()
        if opp_king_square is not None:
            king_neighbors = [sq for sq in chess.SQUARES if chess.square_distance(sq, opp_king_square) == 1]
            reward += 0.03 * sum([self.board.is_attacked_by(self.board.turn, sq) for sq in king_neighbors])

        if self.board.is_check():
            reward += 0.3
            if self.last_check:
                reward += 0.1
            self.last_check = True
        else:
            self.last_check = False

        if self.step_count > 40:
            reward -= 0.01 * (self.step_count - 40)

        return self.get_observation(), reward, done, {}

    def step_with_player(self, action_index, player_id):
        if self.board.turn != (player_id == 0):
            self.board.turn = (player_id == 0)
        return self.step(action_index)

    def count_material(self):
        return sum(PIECE_VALUES[piece.piece_type]
                   for piece in self.board.piece_map().values()
                   if piece.color != self.board.turn and piece.piece_type in PIECE_VALUES)

    def count_pockets(self):
        return sum(self.board.pockets[chess.WHITE].count(pt) + self.board.pockets[chess.BLACK].count(pt)
                   for pt in PIECE_VALUES.keys())

    def get_opponent_king_square(self):
        for square, piece in self.board.piece_map().items():
            if piece.piece_type == chess.KING and piece.color != self.board.turn:
                return square
        return None

    def get_observation(self):
        obs = np.zeros((11, 8, 8), dtype=np.float32)
        piece_map = self.board.piece_map()
        piece_to_idx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

        for square, piece in piece_map.items():
            row = 7 - (square // 8)
            col = square % 8
            idx = piece_to_idx[piece.symbol().upper()]
            obs[idx, row, col] = 1 if piece.color == chess.WHITE else -1

        for i, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]):
            w = self.board.pockets[chess.WHITE].count(pt)
            b = self.board.pockets[chess.BLACK].count(pt)
            v = w - b if self.board.turn == chess.WHITE else b - w
            obs[6 + i, :, :] = v / 5.0

        return obs

    def get_legal_action_indices(self):
        if self.allow_drops:
            return [ALL_POSSIBLE_MOVES.index(move.uci()) for move in self.board.legal_moves]
        else:
            return [
                ALL_POSSIBLE_MOVES.index(move.uci())
                for move in self.board.legal_moves
                if not move.drop
            ]




















































