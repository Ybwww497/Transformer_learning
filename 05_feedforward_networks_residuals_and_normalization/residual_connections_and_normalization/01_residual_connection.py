import torch
import torch.nn as nn

x = torch.randn(2, 10, 64)
linear = nn.Linear(64, 64)

def residual(x):
    fx = linear(x)

    # TODO:实现残差连接
    y = x + fx

    return y

output = residual(x)
print(output.shape)