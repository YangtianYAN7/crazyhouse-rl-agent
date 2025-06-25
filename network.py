import torch
import torch.nn as nn

class ActorCriticNet(nn.Module):
    def __init__(self):
        super(ActorCriticNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_common = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU()
        )
        self.fc_policy = nn.Linear(512, 4672)  # Total number of Crazyhouse possible actions
        self.fc_value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)  # x shape: [batch, 1, 8, 8]
        x = self.flatten(x)
        x = self.fc_common(x)
        policy_logits = self.fc_policy(x)
        value = self.fc_value(x)
        return policy_logits, value



