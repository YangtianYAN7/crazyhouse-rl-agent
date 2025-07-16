# train.py

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic
from action_encoder import ALL_POSSIBLE_MOVES
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CrazyhouseEnv()
input_shape = env.get_observation().shape
n_actions = len(ALL_POSSIBLE_MOVES)

model = ActorCritic(input_shape, n_actions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
writer = SummaryWriter(log_dir="runs")

num_episodes = 10000

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    total_loss = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = model(obs_tensor)

        # 合法动作掩码
        legal_mask = torch.zeros(n_actions).to(device)
        legal_indices = env.get_legal_action_indices()
        legal_mask[legal_indices] = 1

        masked_logits = logits.masked_fill(legal_mask == 0, float('-inf'))
        probs = torch.softmax(masked_logits, dim=-1)

        # 若产生 nan/inf，跳过该步
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("Invalid probs detected, skipping step.")
            break

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_obs, reward, done, info = env.step(action.item())
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        _, next_value = model(next_obs_tensor)

        target = reward + (0.99 * next_value.item() * (1 - int(done)))
        critic_loss = F.mse_loss(value.view(-1), torch.tensor([target], device=device))
        log_prob = torch.log(probs.squeeze(0)[action])
        actor_loss = -log_prob * (target - value.item())
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        total_loss += loss.item()
        obs = next_obs

    print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Total Loss: {total_loss:.4f}")
    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Loss", total_loss, episode)

        # 每 500 局保存一次模型
    if episode % 500 == 0:
        ckpt_path = f"checkpoints/model_ep{episode}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ 已保存中间模型至 {ckpt_path}")

writer.close()
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model.pth")
print("模型已保存至 checkpoints/model.pth")































