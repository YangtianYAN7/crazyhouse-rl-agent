# evaluate.py - self-play 对局 + Elo 评分

import torch
from crazyhouse_env import CrazyhouseEnv
from network import ActorCritic
from action_encoder import ALL_POSSIBLE_MOVES
from elo import update_elo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始 Elo 分数（你也可以从文件中读取）
current_elo = 1000
opponent_elo = 1000

def evaluate_with_elo(model_path, episodes=10):
    global current_elo, opponent_elo

    env = CrazyhouseEnv()
    input_shape = env.get_observation().shape
    n_actions = len(ALL_POSSIBLE_MOVES)

    model = ActorCritic(input_shape, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    opponent_model = ActorCritic(input_shape, n_actions).to(device)
    opponent_model.load_state_dict(model.state_dict())
    opponent_model.eval()

    wins = 0
    draws = 0
    losses = 0

    for i in range(episodes):
        obs = env.reset()
        done = False
        current_player = 0  # 0=model, 1=opponent

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            if current_player == 0:
                with torch.no_grad():
                    logits, _ = model(obs_tensor)
            else:
                with torch.no_grad():
                    logits, _ = opponent_model(obs_tensor)

            legal_mask = torch.zeros(n_actions).to(device)
            legal_indices = env.get_legal_action_indices()
            legal_mask[legal_indices] = 1
            masked_logits = logits.masked_fill(legal_mask == 0, float('-inf'))
            probs = torch.softmax(masked_logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            obs, reward, done, _ = env.step_with_player(action.item(), current_player)
            current_player = 1 - current_player

        # 最终奖励是输赢判断依据
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    win_rate = wins / episodes
    draw_rate = draws / episodes
    result_score = win_rate + 0.5 * draw_rate  # 等效得分

    current_elo, opponent_elo = update_elo(current_elo, opponent_elo, result_score)

    print(f"✅ Evaluation complete: W {wins} / D {draws} / L {losses}")
    print(f"📊 New Elo (Current vs Opponent): {int(current_elo)} vs {int(opponent_elo)}\n")

    return current_elo, opponent_elo





































