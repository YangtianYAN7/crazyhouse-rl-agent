import torch
import numpy as np
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic
from action_encoder import ALL_POSSIBLE_MOVES

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
        return action.item()

def evaluate_with_elo(model_path, episodes=10):
    input_shape = (11, 8, 8)
    n_actions = len(ALL_POSSIBLE_MOVES)

    # 加载被评估模型（最新模型）
    current_model = ActorCritic(input_shape, n_actions).to(device)
    current_model.load_state_dict(torch.load("checkpoints/model.pth", map_location=device))
    current_model.eval()

    # 加载对手模型（支持传入路径或模型对象）
    opponent_model = ActorCritic(input_shape, n_actions).to(device)
    if isinstance(model_path, str):
        opponent_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        opponent_model.load_state_dict(model_path.state_dict())
    opponent_model.eval()

    # 初始化 Elo
    current_elo = 1000
    opponent_elo = 1200

    def update_elo(wins, losses, draws, K=32):
        expected_score = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
        actual_score = (wins + 0.5 * draws) / (wins + losses + draws + 1e-8)
        new_elo = current_elo + K * (actual_score - expected_score)
        return round(new_elo)

    results = []
    rewards = []

    for ep in range(episodes):
        # 黑白交替
        white_first = (ep % 2 == 0)
        env = CrazyhouseEnv(allow_drops=True)
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            legal_actions = env.get_legal_action_indices()
            if not legal_actions:
                break

            if (env.board.turn == white_first):  # 当前模型执子
                action = select_action(current_model, state, legal_actions)
            else:
                action = select_action(opponent_model, state, legal_actions)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        rewards.append(total_reward)

        if reward == 1:
            results.append(1 if env.board.turn != white_first else 0)  # 当前模型赢
        elif reward == -1:
            results.append(0 if env.board.turn != white_first else 1)  # 当前模型输
        else:
            results.append(0.5)

        print(f"✅ Episode {ep + 1}: Reward = {reward:.2f}")

    wins = sum(1 for r in results if r == 1)
    losses = sum(1 for r in results if r == 0)
    draws = sum(1 for r in results if r == 0.5)

    avg_reward = np.mean(rewards)
    new_elo = update_elo(wins, losses, draws)

    print(f"\n🧠 Evaluation complete: {wins} W / {draws} D / {losses} L")
    print(f"🎯 Avg reward = {avg_reward:.2f}")
    print(f"📊 New Elo (Current vs Opponent): {new_elo} vs {opponent_elo}")





































