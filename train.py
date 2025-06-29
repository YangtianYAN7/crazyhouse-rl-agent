import torch
import torch.optim as optim
import torch.nn.functional as F
from crazyhouse_env import CrazyHouseEnv
from network import ActorCriticNet

env = CrazyHouseEnv()
model = ActorCriticNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for episode in range(10):
    obs = env.reset()
    obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for step in range(1):
        logits, value = model(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        next_obs, reward, done, _ = env.step(action.item())
        next_obs_tensor = torch.tensor(next_obs.flatten(), dtype=torch.float32).unsqueeze(0)

        _, next_value = model(next_obs_tensor)

        target = reward + 0.99 * next_value.item() * (1 - int(done))
        advantage = target - value.item()

        policy_loss = -dist.log_prob(action) * advantage
        value_loss = F.mse_loss(value.squeeze(), torch.tensor([target]))

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        if done:
            break

    print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f}")










