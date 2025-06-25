import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, input_channels=1, board_size=8, action_size=4672):
        super(PolicyValueNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        flat_size = 128 * board_size * board_size  # 128x8x8 = 8192

        # policy head
        self.policy_fc = nn.Linear(flat_size, action_size)

        # value head
        self.value_fc1 = nn.Linear(flat_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 输入形状 x: [batch_size, 1, 8, 8]
        x = F.relu(self.conv1(x))  # -> [B, 32, 8, 8]
        x = F.relu(self.conv2(x))  # -> [B, 64, 8, 8]
        x = F.relu(self.conv3(x))  # -> [B, 128, 8, 8]
        x = x.view(x.size(0), -1)  # flatten -> [B, 8192]

        # policy 分支
        policy_logits = self.policy_fc(x)  # -> [B, action_size]

        # value 分支
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)  # -> [B, 1]

        return policy_logits, value


