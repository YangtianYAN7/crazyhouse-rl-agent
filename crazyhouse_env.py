import numpy as np
import chess
import chess.variant
import chess.svg
import cairosvg
from io import BytesIO
from action_encoder import decode_action, encode_action
import gym

class CrazyhouseEnv(gym.Env):
    def __init__(self):
        self.board = chess.variant.CrazyhouseBoard()
        self.max_steps = 50
        self.steps = 0

    def reset(self):
        self.board.reset()
        self.steps = 0
        return self.get_observation()

    def get_observation(self):
        board_str = self.board.fen()
        obs = np.zeros(773, dtype=np.float32)
        obs[:len(board_str)] = [ord(c) for c in board_str]
        return obs

    def step(self, action_idx):
        self.steps += 1
        action_uci = decode_action(action_idx)
        reward = -1.0
        done = False

        if action_uci in [move.uci() for move in self.board.legal_moves]:
            self.board.push_uci(action_uci)
            reward = 1.0
        else:
            done = True

        if self.board.is_game_over() or self.steps >= self.max_steps:
            done = True

        return self.get_observation(), reward, done, {}

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            svg = chess.svg.board(self.board)
            png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
            image = BytesIO(png_bytes).getvalue()
            import PIL.Image
            img = PIL.Image.open(BytesIO(image)).convert("RGB")
            return np.array(img)
        else:
            print(self.board)

    def close(self):
        pass


















