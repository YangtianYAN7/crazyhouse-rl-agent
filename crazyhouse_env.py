import gym
import numpy as np
from gym import spaces

PIECES = ['P', 'N', 'B', 'R', 'Q', 'K']  
PIECE_TO_ID = {p: i+1 for i, p in enumerate(PIECES)}  
PIECE_TO_ID.update({p.lower(): -(i+1) for i, p in enumerate(PIECES)})  

class CrazyHouseEnv(gym.Env):
    def __init__(self):
        super(CrazyHouseEnv, self).__init__()

        self.board = np.zeros((8, 8), dtype=np.int8)
        self._initialize_board()  

        self.inventory = {
            'white': {p: 0 for p in PIECES},
            'black': {p: 0 for p in PIECES}
        }

        self.observation_space = spaces.Box(low=-6, high=6, shape=(8, 8), dtype=np.int8)
        self.action_space = spaces.Discrete(4672)  

        self.current_player = 'white'
        self.done = False

    def _initialize_board(self):
 
        self.board[0] = [PIECE_TO_ID[p.lower()] for p in ['R','N','B','Q','K','B','N','R']]
        self.board[1] = [PIECE_TO_ID['p'] for _ in range(8)]

        self.board[6] = [PIECE_TO_ID['P'] for _ in range(8)]
        self.board[7] = [PIECE_TO_ID[p] for p in ['R','N','B','Q','K','B','N','R']]

    def reset(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self._initialize_board()
        self.inventory = {
            'white': {p: 0 for p in PIECES},
            'black': {p: 0 for p in PIECES}
        }
        self.current_player = 'white'
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return self.board.copy()

    def step(self, action):
        if isinstance(action, tuple) and action[0] == 'move':
            _, from_sq, to_sq = action
            fx, fy = from_sq
            tx, ty = to_sq
            piece = self.board[fx, fy]
            captured = self.board[tx, ty]

            if piece == 0 or (self.current_player == 'white' and piece < 0) or (self.current_player == 'black' and piece > 0):
                return self._get_obs(), -1.0, False, {'error': 'Invalid move source'}

            self.board[tx, ty] = piece
            self.board[fx, fy] = 0

            if captured != 0:
                captured_piece = PIECES[abs(captured) - 1]
                if self.current_player == 'white':
                    self.inventory['white'][captured_piece] += 1
                else:
                    self.inventory['black'][captured_piece] += 1

        elif isinstance(action, tuple) and action[0] == 'drop':
            _, piece_type, target_sq = action
            tx, ty = target_sq

            if self.board[tx, ty] != 0:
                return self._get_obs(), -1.0, False, {'error': 'Drop target not empty'}

            if self.inventory[self.current_player][piece_type] <= 0:
                return self._get_obs(), -1.0, False, {'error': 'No piece to drop'}

            piece_id = PIECE_TO_ID[piece_type] if self.current_player == 'white' else PIECE_TO_ID[piece_type.lower()]
            self.board[tx, ty] = piece_id
            self.inventory[self.current_player][piece_type] -= 1

        else:
            return self._get_obs(), -1.0, False, {'error': 'Unknown action type'}

        self.current_player = 'black' if self.current_player == 'white' else 'white'
        return self._get_obs(), 0.0, self.done, {}

    def render(self, mode='human'):
        print("Current board:")
        print(self.board)
        print(f"{self.current_player} to move")
        print(f"Inventory: {self.inventory[self.current_player]}")

    def legal_actions(self):
        actions = []
        color = 1 if self.current_player == 'white' else -1

        for x in range(8):
            for y in range(8):
                piece = self.board[x, y]
                if (color == 1 and piece > 0) or (color == -1 and piece < 0):

                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < 8 and 0 <= ny < 8:
                            target = self.board[nx, ny]
                            if (color == 1 and target <= 0) or (color == -1 and target >= 0):
                                actions.append(('move', (x, y), (nx, ny)))


        for piece in PIECES:
            if self.inventory[self.current_player][piece] > 0:
                for x in range(8):
                    for y in range(8):
                        if self.board[x, y] == 0:
                            actions.append(('drop', piece, (x, y)))

        return actions

if __name__ == "__main__":
    env = CrazyHouseEnv()
    obs = env.reset()
    env.render()
    actions = env.legal_actions()
    print(f"Legal actions: {actions[:5]} (total {len(actions)})")
    obs, reward, done, info = env.step(actions[0])
    env.render()
