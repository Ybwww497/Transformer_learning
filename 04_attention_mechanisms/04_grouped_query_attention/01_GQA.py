import torch
import math

Q = torch.randn(4, 32, 256, 128)
K = torch.randn(4, 8, 256, 128)
V = torch.randn(4, 8, 256,128)

# 1. 扩展KV Head
K_expand = K.repeat_interleave(4, dim=1)
V_expand = V.repeat_interleave(4, dim=1)

# 2. 计算Scaled Dot-Product Attention 分数
scores = Q @ K_expand.transpose(-2, -1) / math.sqrt(Q.shape[-1])   # `Q.shape[-1]` 取的是最后一维的长度，`Q[-1]` 取的是 Q 沿第一维的最后一项数据。

# 3. Softmax
weights = torch.softmax(scores, dim=-1)

# 4. 对V进行加权求和
out = weights @ V_expand

print(out.shape)