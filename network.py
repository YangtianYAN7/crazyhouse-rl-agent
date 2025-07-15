import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()

        self.flatten = nn.Flatten()
        flattened_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.shared = nn.Sequential(
            self.flatten,
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        #print("DEBUG: input shape:", x.shape)  # 调试用，训练完成后可删除
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value













