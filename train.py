import torch
import torch.nn as nn
import torch.optim as optim
from network import ActorCritic
from crazyhouse_env import CrazyhouseEnv
from action_encoder import ALL_POSSIBLE_MOVES
from evaluate import evaluate_with_elo
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(model, state, legal_actions):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        logits, _ = model(state)
        mask = torch.zeros(logits.shape[-1], device=device)
        mask[legal_actions] = 1
        masked_logits = logits + (mask + 1e-8).log()
        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def train():
    input_shape = (11, 8, 8)
    n_actions = len(ALL_POSSIBLE_MOVES)
    model = ActorCritic(input_shape, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_episodes = 500
    env = CrazyhouseEnv(allow_drops=False)  # ⛔ Phase 1: 禁用落子
    switch_phase_at = 200  # 🚀 200局后切换到 Crazyhouse 模式

    os.makedirs("checkpoints", exist_ok=True)

    for episode in range(num_episodes):
        if episode == switch_phase_at:
            print("🚀 进入 Phase 2：启用 Crazyhouse 落子")
            env.allow_drops = True

        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        total_reward = 0
        total_loss = 0

        for t in range(60):
            legal_actions = env.get_legal_action_indices()
            if not legal_actions:
                break
            action, log_prob = select_action(model, state, legal_actions)
            _, value = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            total_reward += reward

            state = next_state
            if done:
                break

        returns = compute_returns(rewards)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.cat(values).squeeze()
        log_probs = torch.stack(log_probs)

        advantage = returns - values
        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = loss.item()
        phase = "Crazyhouse" if env.allow_drops else "Classic"
        print(f"🎯 Episode {episode} | Phase: {phase} | Reward: {total_reward:.2f} | Loss: {total_loss:.4f}")

        # 每5轮保存模型并进行 Elo 评估
        if episode % 5 == 0:
            checkpoint_path = f"checkpoints/epoch{episode}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"📦 Saved model at {checkpoint_path}")

            print(f"🧠 Evaluating model at epoch {episode} with Elo...")
            evaluate_with_elo(model, episodes=10)

            # 另存一份当前对手用模型
            torch.save(model.state_dict(), "checkpoints/model.pth")
            print(f"✅ 模型已保存至 checkpoints/model.pth")

if __name__ == "__main__":
    train()




























































