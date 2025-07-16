# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        c, h, w = input_shape

        self.shared = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),  # 输出 [32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 输出 [64, 8, 8]
            nn.ReLU(),
            nn.Flatten(),                                # 输出 64*8*8 = 4096
            nn.Linear(4096, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value























