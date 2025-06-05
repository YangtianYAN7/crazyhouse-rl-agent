import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from network import PolicyValueNet
from crazyhouse_env import CrazyHouseEnv
from action_encoder import encode_action, legal_action_indices

NUM_EPISODES = 1000
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SAVE_EVERY = 100
MAX_STEPS_PER_EPISODE = 100

def encode_state(board):
    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)  
    inventory = torch.zeros((1, 8, 8))  
    meta = torch.zeros((1, 8, 8))       
    state_tensor = torch.cat([board_tensor, inventory, meta], dim=0)  
    return state_tensor

def sample_action(actions):
    return random.choice(actions) if actions else None

def compute_policy_loss(predicted_log_probs, action_index):
    target = torch.tensor([action_index], dtype=torch.long)
    loss = nn.NLLLoss()(predicted_log_probs, target)
    return loss

def compute_value_loss(predicted_value, target_value):
    return nn.MSELoss()(predicted_value, torch.tensor([[target_value]]))

def train():
    env = CrazyHouseEnv()
    model = PolicyValueNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for episode in range(NUM_EPISODES):
        print(f"[Episode {episode+1}] Starting...")
        obs = env.reset()
        done = False
        step_count = 0
        states, action_idxs, rewards = [], [], []

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            actions = env.legal_actions()
            if not actions:
                print("  No legal action, breaking.")
                break

            action = sample_action(actions)
            action_idx = encode_action(action)
            if action_idx == -1:
                print(f"  Invalid action: {action}, skipping.")
                continue

            print(f"  Step {step_count+1}: Action taken: {action} → idx {action_idx}")

            state_tensor = encode_state(obs)
            states.append(state_tensor)
            action_idxs.append(action_idx)

            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            step_count += 1

        final_value = sum(rewards)

        model.train()
        for i in range(len(states)):
            state = states[i].unsqueeze(0)  # [1,3,8,8]
            policy_log_probs, value_pred = model(state)
            loss_p = compute_policy_loss(policy_log_probs, action_idxs[i])
            loss_v = compute_value_loss(value_pred, final_value)
            loss = loss_p + loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"  Train Step {i+1}: policy_loss={loss_p.item():.4f}, value_loss={loss_v.item():.4f}, total_loss={loss.item():.4f}")

        if (episode + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"model_ep{episode+1}.pt")
            print(f"[Episode {episode+1}] Model saved. Total reward: {final_value:.2f}\n")

if __name__ == "__main__":
    train()


