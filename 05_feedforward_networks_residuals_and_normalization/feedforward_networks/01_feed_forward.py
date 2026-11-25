import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, C):
        super().__init__()

        # TODO 1:Linear, C -> 4*C
        self.fc1 = nn.Linear(C, 4*C)

        # TODO 2:GELU激活函数
        self.act = nn.GELU()

        # TODO 3:Linear, 4*C -> C
        self.fc2 = nn.Linear(4*C, C)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x 

x = torch.randn(2, 4, 8)

ffn = FeedForward(8)
output = ffn.forward(x)
print("input.shape =", x.shape)
print("output.shape =", output.shape)
