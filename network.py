import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim=773, hidden_dim=512, action_dim=4672):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)










