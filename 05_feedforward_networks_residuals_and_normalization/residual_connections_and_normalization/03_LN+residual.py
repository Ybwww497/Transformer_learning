import torch
import torch.nn as nn

x = torch.randn(2, 10, 64)

linear = nn.Linear(64, 64)
ln = nn.LayerNorm(64)

def forward(x):
    fx = linear(x)

    # 1. 实现残差连接
    residual = fx + x

    # 2. 实现LayerNorm
    y = ln(residual)

    return y

output = forward(x)
print("output.shape =", output.shape)