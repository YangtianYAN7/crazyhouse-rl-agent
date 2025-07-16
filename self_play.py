# self_play.py

import torch
import numpy as np
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic
from action_encoder import ALL_POSSIBLE_MOVES
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(model, obs, legal_indices):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    logits, _ = model(obs_tensor)
    mask = torch.zeros(len(ALL_POSSIBLE_MOVES)).to(device)
    mask[legal_indices] = 1
    masked_logits = logits.masked_fill(mask == 0, float('-inf'))
    probs = torch.softmax(masked_logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    return dist.sample().item()

def self_play(model_path_1, model_path_2=None, games=20):
    env = CrazyhouseEnv()
    input_shape = env.get_observation().shape
    n_actions = len(ALL_POSSIBLE_MOVES)

    model1 = ActorCritic(input_shape, n_actions).to(device)
    model1.load_state_dict(torch.load(model_path_1))
    model1.eval()

    model2 = model1 if model_path_2 is None else ActorCritic(input_shape, n_actions).to(device)
    if model_path_2:
        model2.load_state_dict(torch.load(model_path_2))
        model2.eval()

    wins_model1 = 0
    wins_model2 = 0
    draws = 0

    for game in range(games):
        obs = env.reset()
        done = False
        turn = 0

        while not done:
            legal_indices = env.get_legal_action_indices()
            if turn % 2 == 0:
                action = select_action(model1, obs, legal_indices)
            else:
                action = select_action(model2, obs, legal_indices)
            obs, reward, done, _ = env.step(action)
            turn += 1

        if reward == 1.0:
            winner = "model1" if turn % 2 == 1 else "model2"
        elif reward == -1.0:
            winner = "model2" if turn % 2 == 1 else "model1"
        else:
            winner = "draw"

        if winner == "model1":
            wins_model1 += 1
        elif winner == "model2":
            wins_model2 += 1
        else:
            draws += 1

        print(f"Game {game+1}: Winner = {winner}")

    print("\n🎮 自我对弈结果：")
    print(f"model1 胜：{wins_model1} | model2 胜：{wins_model2} | 平局：{draws}")
    return wins_model1, wins_model2, draws

if __name__ == "__main__":
    self_play("checkpoints/model.pth")  # 如果不传第二个模型，就是 self-vs-self

































