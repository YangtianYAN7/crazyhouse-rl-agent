import chess.variant
import chess.pgn
import random
import os

from crazyhouse_env import CrazyhouseEnv


def self_play(model, action_encoder, episodes=1, temperature=1.0, save_dir="games"):
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(episodes):
        env = CrazyhouseEnv()
        obs = env.reset()
        done = False
        moves = []
        boards = [env.board.copy()]

        while not done:
            legal_indices = env.get_legal_action_indices()
            if not legal_indices:
                break

            logits, _ = model(obs.unsqueeze(0))
            probs = logits.softmax(dim=-1).squeeze().detach().numpy()

            # 只在合法动作中采样
            masked_probs = probs[legal_indices]
            masked_probs = masked_probs / masked_probs.sum()
            move_index = random.choices(legal_indices, weights=masked_probs, k=1)[0]

            action = action_encoder.decode(move_index)
            moves.append(action)
            obs, reward, done, _ = env.step(action)
            boards.append(env.board.copy())

        # 保存 PGN 棋谱
        game = chess.pgn.Game()
        game.headers["Event"] = "Self-Play"
        game.headers["Result"] = env.board.result()
        node = game

        for move in moves:
            node = node.add_main_variation(chess.Move.from_uci(move))

        pgn_path = os.path.join(save_dir, f"game_ep{ep}.pgn")
        with open(pgn_path, "w") as f:
            print(game, file=f)

        print(f"✅ 已保存 PGN 棋谱到 {pgn_path}")




































