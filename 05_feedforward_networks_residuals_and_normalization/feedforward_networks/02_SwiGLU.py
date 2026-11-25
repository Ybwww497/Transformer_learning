import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # 1. Gate 分支经过SiLU, 再与Up分支逐元素相乘
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)

        # 2. 将hidden映射回原来的维度
        out = self.down_proj(hidden)

        return out

model = SwiGLU(dim=512, hidden_dim=1536)

x = torch.randn(2, 128, 512)
y = model(x)

print(y.shape)