import torch
import torch.optim as optim
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CrazyhouseEnv()
model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter("runs/crazyhouse_experiment")
os.makedirs("checkpoints", exist_ok=True)

EPISODES = 500
STEPS_PER_EPISODE = 50

for episode in range(EPISODES):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
    total_reward = 0
    total_loss = 0

    for step in range(STEPS_PER_EPISODE):
        logits, value = model(obs)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1).item()

        next_obs, reward, done, info = env.step(action)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device).unsqueeze(0)
        _, next_value = model(next_obs)

        target = reward + (0.99 * next_value.item() * (1 - int(done)))
        advantage = target - value.item()

        actor_loss = -torch.log(probs[0, action]) * advantage
        critic_loss = torch.nn.functional.mse_loss(value.squeeze(), torch.tensor([target]).to(device))
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        total_loss += loss.item()
        obs = next_obs

        if done:
            break

    writer.add_scalar("Reward/Episode", total_reward, episode)
    writer.add_scalar("Loss/Episode", total_loss, episode)

    if episode % 50 == 0:
        board_image = env.render(mode="rgb_array")
        if board_image is not None:
            writer.add_image("Board/Image", torch.tensor(board_image).permute(2, 0, 1), episode)

    # ✅ 模型保存逻辑
    if (episode + 1) % 100 == 0 or episode == EPISODES - 1:
        save_path = f"checkpoints/model_ep{episode+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved at episode {episode+1} → {save_path}")

    print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Total Loss: {total_loss:.4f}")

writer.close()





















