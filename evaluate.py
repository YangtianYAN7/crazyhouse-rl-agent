# evaluate.py - 对比两个模型 ELO 水平（胜率估计）

import torch
import numpy as np
from crazyhouse_env import CrazyhouseEnv
from action_encoder import encode_action_index
from network import ChessNet

def evaluate(model_a, model_b, episodes=20, device='cpu'):
    model_a.eval()
    model_b.eval()
    model_a = model_a.to(device)
    model_b = model_b.to(device)

    win_a, win_b, draw = 0, 0, 0

    for ep in range(episodes):
        env = CrazyhouseEnv()
        state = env.reset()
        done = False
        turn = 0

        while not done:
            model = model_a if turn % 2 == 0 else model_b
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _ = model(state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            legal_indices = env.get_legal_action_indices()
            probs_masked = np.zeros_like(probs)
            probs_masked[legal_indices] = probs[legal_indices]
            if probs_masked.sum() == 0:
                action = np.random.choice(legal_indices)
            else:
                probs_masked /= probs_masked.sum()
                action = np.random.choice(len(probs), p=probs_masked)

            state, reward, done, _ = env.step(action)
            turn += 1

        if reward == 1.0:
            if turn % 2 == 1:
                win_a += 1
            else:
                win_b += 1
        elif reward == 0.0:
            draw += 1
        else:
            if turn % 2 == 1:
                win_b += 1
            else:
                win_a += 1

    print(f"✅ Evaluation over {episodes} games:")
    print(f"Model A wins: {win_a}, Model B wins: {win_b}, Draws: {draw}")
    win_rate = win_a / (win_a + win_b + draw)
    print(f"Estimated Elo diff: {400 * np.log10((win_rate + 1e-5) / (1 - win_rate + 1e-5)):.2f}")
































