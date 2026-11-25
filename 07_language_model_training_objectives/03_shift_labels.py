#=============shift labels(标签错位)=============
import torch

# [B, T]
tokens = torch.tensor([
    [10, 20, 30, 40, 50],
    [11, 21, 31, 41, 51]
])

# 输入：去掉最后一个token
x = tokens[:,:-1]

# 标签：去掉第一个token
y = tokens[:, 1:]

print("x =", x)
print("y =", y)
print("x.shape =", x.shape)
print("y.shape =", y.shape)
