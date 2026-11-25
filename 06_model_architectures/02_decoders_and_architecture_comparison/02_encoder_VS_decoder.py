import torch

B = 2
T_src = 10
T_tgt = 6
H = 4
D = 8

q = torch.randn(B, T_tgt, H * D)
k = torch.randn(B, T_src, H * D)

# 任务1：将 q 拆分成 [B, H, T_tgt, D]
q = q.reshape(B, T_tgt, H , D).transpose(1, 2)

# 任务2：将 k 拆分成 [B, H, T_src, D]
k = k.reshape(B, T_src, H , D).transpose(1, 2)

# 任务3：计算 Attention Scores
scores = q @ k.transpose(-1, -2)

print("Q:", q.shape)
print("K:", k.shape)
print("Scores:", scores.shape)