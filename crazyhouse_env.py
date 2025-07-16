# crazyhouse_env.py

import chess
import numpy as np
from action_encoder import encode_action, decode_action, ALL_POSSIBLE_MOVES

class CrazyhouseEnv:
    def __init__(self):
        self.board = chess.Board()
        self.update_action_space()
        self.observation_space = type('', (), {})()
        self.observation_space.shape = (6, 8, 8)
        self.step_count = 0  # 加入最大步数限制

    def update_action_space(self):
        self.action_space = type('', (), {})()
        self.action_space.n = len(ALL_POSSIBLE_MOVES)

    def reset(self):
        self.board.reset()
        self.step_count = 0
        self.update_action_space()
        return self.get_observation()

    def step(self, action_index):
        move = decode_action(self.board, action_index)
        prev_material = self.count_material()

        if move not in self.board.legal_moves:
            return self.get_observation(), -1.0, True, {}

        self.board.push(move)
        self.step_count += 1
        self.update_action_space()

        reward = 0.0
        done = False

        # 终止条件：强制最大步数
        if self.step_count >= 50:
            return self.get_observation(), -1.0, True, {}

        # 奖励机制
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
            reward = 0.005  # 小鼓励（已削弱）

            # 吃子奖励
            new_material = self.count_material()
            captured = prev_material - new_material
            if captured > 0:
                reward += 0.5 * captured  # 放大吃子影响

            # 控制中心格奖励
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            reward += 0.05 * sum([self.board.is_attacked_by(self.board.turn, sq) for sq in center_squares])

            # 被将惩罚
            if self.board.is_check():
                reward -= 0.3

        return self.get_observation(), reward, done, {}

    def count_material(self):
        return sum(piece.piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                        chess.ROOK, chess.QUEEN]
                   for piece in self.board.piece_map().values()
                   if piece.color != self.board.turn)

    def get_observation(self):
        piece_map = self.board.piece_map()
        obs = np.zeros((6, 8, 8), dtype=np.float32)
        piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}

        for square, piece in piece_map.items():
            row = 7 - (square // 8)
            col = square % 8
            idx = piece_to_index[piece.symbol().upper()]
            obs[idx, row, col] = 1 if piece.color == chess.WHITE else -1

        return obs

    def get_legal_action_indices(self):
        legal_moves = list(self.board.legal_moves)
        return [ALL_POSSIBLE_MOVES.index(m.uci()) for m in legal_moves]




























