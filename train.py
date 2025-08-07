# train.py - 引入 self-play 机制进行强化学习训练

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic
from action_encoder import ALL_POSSIBLE_MOVES
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化环境和模型
env = CrazyhouseEnv()
input_shape = env.get_observation().shape
n_actions = len(ALL_POSSIBLE_MOVES)

# 当前训练模型（玩家1）
model = ActorCritic(input_shape, n_actions).to(device)

# 固定对手模型（玩家2）
opponent_model = ActorCritic(input_shape, n_actions).to(device)
opponent_model.load_state_dict(model.state_dict())  # 初始化为同一模型
opponent_model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

os.makedirs("checkpoints", exist_ok=True)
writer = SummaryWriter(log_dir="runs")

num_episodes = 500
save_opponent_every = 20  # 每隔N局，将当前模型保存为对手

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    current_player = 0  # 0 = model, 1 = opponent
    total_reward = 0
    total_loss = 0
    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy = 0

    if episode == 0:
        with open("model_structure.txt", "w") as f:
            f.write(str(model))

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

        # 玩家0：我们训练的模型
        if current_player == 0:
            logits, value = model(obs_tensor)
        else:
            with torch.no_grad():
                logits, _ = opponent_model(obs_tensor)

        # 合法动作 mask
        legal_mask = torch.zeros(n_actions).to(device)
        legal_indices = env.get_legal_action_indices()
        legal_mask[legal_indices] = 1
        masked_logits = logits.masked_fill(legal_mask == 0, float('-inf'))
        probs = torch.softmax(masked_logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("⚠️ Invalid probs detected, skipping step.")
            break

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # 进行一步 self-play 对弈
        next_obs, reward, done, info = env.step_with_player(action.item(), current_player)

        # 只训练玩家0的模型
        if current_player == 0:
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, next_value = model(next_obs_tensor)

            target = reward + (0.99 * next_value.item() * (1 - int(done)))
            critic_loss = F.mse_loss(value.view(-1), torch.tensor([target], device=device))
            log_prob = torch.log(probs.squeeze(0)[action])
            actor_loss = -log_prob * (target - value.item())
            entropy = dist.entropy()

            loss = actor_loss + critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            total_reward += reward

        obs = next_obs
        current_player = 1 - current_player  # 交换下棋方

    # 日志输出
    print(f"🎯 Episode {episode} | Reward: {total_reward:.2f} | Loss: {total_loss:.4f}")

    writer.add_scalar("Reward", total_reward, episode)
    writer.add_scalar("Loss", total_loss, episode)
    writer.add_scalar("ActorLoss", total_actor_loss, episode)
    writer.add_scalar("CriticLoss", total_critic_loss, episode)
    writer.add_scalar("Entropy", total_entropy, episode)

    # 每 20 局更新对手模型
    if (episode + 1) % save_opponent_every == 0:
        opponent_model.load_state_dict(model.state_dict())

    # 每 100 局打印 top-5 策略
    if episode % 100 == 0:
        topk = torch.topk(probs.squeeze(0), 5)
        print("🧠 Top-5 Actions:")
        for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
            print(f"  {ALL_POSSIBLE_MOVES[idx]}: {val:.4f}")

    # 每 500 局保存一次
    if episode % 500 == 0:
        ckpt_path = f"checkpoints/model_ep{episode}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ 已保存模型至 {ckpt_path}")

    # 每 5 局保存并评估
    if (episode + 1) % 5 == 0:
        model_path = f"checkpoints/epoch{episode+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"📦 Saved model at {model_path}")
        print(f"🧪 Evaluating model at epoch {episode+1}...")
        evaluate(model_path, episodes=5)

# 最终保存
torch.save(model.state_dict(), "checkpoints/model.pth")
writer.close()
print("✅ 模型已保存至 checkpoints/model.pth")



























































