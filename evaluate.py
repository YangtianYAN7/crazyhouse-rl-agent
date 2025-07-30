import torch
import numpy as np
import chess
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic
from action_encoder import ALL_POSSIBLE_MOVES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构造反向映射：从 move str -> index
MOVE_TO_INDEX = {move: idx for idx, move in enumerate(ALL_POSSIBLE_MOVES)}

def evaluate(model_path, episodes=10, verbose=True):
    env = CrazyhouseEnv()
    input_shape = env.get_observation().shape
    action_size = len(ALL_POSSIBLE_MOVES)

    model = ActorCritic(input_shape, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    wins = 0
    total_reward = 0

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(obs_tensor)

                # 添加合法动作 mask
                legal_indices = env.get_legal_action_indices()
                legal_mask = torch.zeros(action_size).to(device)
                legal_mask[legal_indices] = 1

                masked_logits = logits.masked_fill(legal_mask == 0, float('-inf'))
                probs = torch.softmax(masked_logits, dim=-1)
                action_index = torch.argmax(probs, dim=-1).item()

            obs, reward, done, _ = env.step(action_index)
            ep_reward += reward

        if ep_reward > 0:
            wins += 1
        total_reward += ep_reward

        if verbose:
            print(f"🎯 Episode {ep + 1}: Reward = {ep_reward:.2f}")

    win_rate = wins / episodes
    avg_reward = total_reward / episodes
    print(f"✅ Evaluation complete: Win rate = {win_rate * 100:.1f}%, Avg reward = {avg_reward:.2f}")
    return win_rate, avg_reward




































