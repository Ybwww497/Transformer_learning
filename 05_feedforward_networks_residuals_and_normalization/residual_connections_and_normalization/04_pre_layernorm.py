import torch
import torch.nn as nn

x = torch.randn(2, 10, 64)

linear = nn.Linear(64, 64)
ln = nn.LayerNorm(64)

def forward(x):
    # 1. 第一步：Pre-LayerNorm
    ln_x = ln(x)

    # 2. 第二步：Linear
    Linear_ln = linear(ln_x)

    # 3. 第三步：残差连接
    y = Linear_ln + x

    return y

output = forward(x)
print("output.shape =", output.shape)