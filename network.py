import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, action_size=4672):
        super(ActorCriticNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value




