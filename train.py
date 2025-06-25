import torch
import torch.nn as nn
import torch.optim as optim
from crazyhouse_env import CrazyHouseEnv
from network import ActorCriticNet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化环境和模型
env = CrazyHouseEnv()
model = ActorCriticNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

max_episodes = 10

for episode in range(max_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 8, 8]
        policy_logits, value = model(obs_tensor)

        legal_actions = env.legal_actions()
        legal_actions_tensor = torch.tensor(legal_actions, dtype=torch.long, device=device)

        action_probs = torch.softmax(policy_logits, dim=-1).squeeze(0)
        legal_probs = action_probs[legal_actions_tensor]
        legal_probs /= legal_probs.sum() + 1e-8

        action_idx = torch.multinomial(legal_probs, num_samples=1).item()
        action = legal_actions[action_idx]

        next_obs, reward, done, _ = env.step(action)
        total_reward += reward

        # 计算损失
        _, next_value = model(torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
        advantage = reward + (0.99 * next_value.item() * (1 - int(done))) - value.item()

        policy_loss = -torch.log(action_probs[action]) * advantage
        value_loss = advantage ** 2
        loss = policy_loss + value_loss

        # 更新网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs

    print(f"[Episode {episode + 1}] Total Reward: {total_reward:.2f}")







