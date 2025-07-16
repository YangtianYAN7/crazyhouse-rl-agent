# evaluate.py

import torch
import numpy as np
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic
from action_encoder import ALL_POSSIBLE_MOVES
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_model_action(model, obs, legal_indices):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    logits, _ = model(obs_tensor)

    # 合法掩码
    legal_mask = torch.zeros(len(ALL_POSSIBLE_MOVES)).to(device)
    legal_mask[legal_indices] = 1
    masked_logits = logits.masked_fill(legal_mask == 0, float('-inf'))

    probs = torch.softmax(masked_logits, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        return random.choice(legal_indices)  # 防御性 fallback
    dist = torch.distributions.Categorical(probs)
    return dist.sample().item()

def select_random_action(legal_indices):
    return random.choice(legal_indices)

def evaluate_model(model_path, num_games=50):
    env = CrazyhouseEnv()
    input_shape = env.get_observation().shape
    n_actions = len(ALL_POSSIBLE_MOVES)

    model = ActorCritic(input_shape, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    model_elo = 1000
    random_elo = 1000
    K = 32

    for game in range(num_games):
        obs = env.reset()
        done = False
        turn = 0  # 偶数：模型先手，奇数：随机先手

        while not done:
            legal_indices = env.get_legal_action_indices()
            if turn % 2 == 0:
                action = select_model_action(model, obs, legal_indices)
            else:
                action = select_random_action(legal_indices)

            obs, reward, done, _ = env.step(action)
            turn += 1

        # 计算 Elo 更新（模型是先手）
        if reward == 1.0 and turn % 2 == 1:
            S_model = 1  # 模型赢
        elif reward == -1.0 and turn % 2 == 1:
            S_model = 0  # 模型输
        else:
            S_model = 0.5  # 平局或无效

        E_model = 1 / (1 + 10 ** ((random_elo - model_elo) / 400))
        model_elo += K * (S_model - E_model)
        random_elo += K * ((1 - S_model) - (1 - E_model))

        print(f"Game {game + 1} - Result: {'Win' if S_model == 1 else 'Draw' if S_model == 0.5 else 'Loss'} | Model Elo: {round(model_elo)}")

    print("\n🎯 评估完成！")
    print(f"📈 模型估计 Elo: {round(model_elo)}")
    print(f"🎲 随机策略 Elo: {round(random_elo)}")

if __name__ == "__main__":
    evaluate_model("checkpoints/model.pth", num_games=50)































