import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CrazyhouseEnv()

# 正确获取模型输入维度和动作维度
input_shape = env.get_observation().shape
n_actions = env.action_space.n

model = ActorCritic(input_shape, n_actions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 设置 TensorBoard 日志
writer = SummaryWriter(log_dir="runs")

# 设置训练轮数
num_episodes = 100  # 你当前要求训练 10 轮

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    total_loss = 0
    step = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        logits, value = model(obs_tensor)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_obs, reward, done, info = env.step(action.item())

        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
        _, next_value = model(next_obs_tensor)

        target = reward + (0.99 * next_value.item() * (1 - int(done)))
        critic_loss = F.mse_loss(value.squeeze(), torch.tensor([target]).to(device))

        log_prob = torch.log(probs.squeeze(0)[action])
        actor_loss = -log_prob * (target - value.item())

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        total_loss += loss.item()
        obs = next_obs
        step += 1

    # 输出每一轮的训练结果
    print(f"Episode {episode} | Total Reward: {total_reward:.2f} | Total Loss: {total_loss:.4f}")

    # 写入 TensorBoard
    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Loss", total_loss, episode)

# 关闭日志记录
writer.close()

# 保存模型
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model.pth")
print("模型已保存至 checkpoints/model.pth")






















