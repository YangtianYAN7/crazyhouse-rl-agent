import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, board_height=8, board_width=8, inventory_channels=12):
        super(PolicyValueNet, self).__init__()

        self.input_channels = 3
        self.board_height = board_height
        self.board_width = board_width

        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_height * board_width, board_height * board_width * 12) 

        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_height * board_width, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)  

        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  

        return policy, value

def test():
    net = PolicyValueNet()
    dummy_input = torch.randn(1, 3, 8, 8)  
    policy, value = net(dummy_input)
    print("Policy logits shape:", policy.shape)  
    print("Value shape:", value.shape)            

if __name__ == '__main__':
    test()
