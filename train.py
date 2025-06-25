import torch
import torch.nn.functional as F
import torch.optim as optim
from crazyhouse_env import CrazyHouseEnv
from network import PolicyValueNet
import numpy as np
import random
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CrazyHouseEnv()
model = PolicyValueNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

gamma = 0.99
max_episodes = 1000
train_step = 0

for episode in range(max_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,8,8]
        policy_logits, value = model(obs_tensor)

        # 获取合法动作
        legal_actions = env.legal_actions()

        if not legal_actions:
            logging.warning("No legal actions available. Skipping.")
            break

        # 策略 mask，非法动作设置为 -inf
        logits = policy_logits.detach().cpu().numpy().flatten()
        mask = np.full_like(logits, -np.inf)
        for a in legal_actions:
            mask[a] = logits[a]
        masked_logits = torch.tensor(mask).to(device)
        probs = F.softmax(masked_logits, dim=0)

        action = torch.multinomial(probs, num_samples=1).item()
        log_prob = torch.log(probs[action] + 1e-8)  # 防止 log(0)

        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        # Advantage 估计
        advantage = reward - value.item()

        # 损失函数
        policy_loss = -log_prob * advantage
        value_loss = F.mse_loss(value, torch.tensor([[reward]], dtype=torch.float32).to(device))
        total_loss = policy_loss + value_loss

        # 优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 日志输出
        logging.debug(f"Step {train_step}: reward={reward:.1f}, value={value.item():.4f}, advantage={advantage:.4f}")
        logging.debug(f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}, total_loss={total_loss.item():.4f}")

        train_step += 1
        obs = next_obs

    logging.info(f"Episode {episode + 1} finished, total_reward = {episode_reward:.2f}")




