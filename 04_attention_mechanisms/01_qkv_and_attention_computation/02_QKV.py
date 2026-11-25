import torch
import torch.nn as nn

B, T, C = 2, 5, 8
D = 4

X = torch.randn(B, T, C)

q_proj = nn.Linear(C, C)
k_proj = nn.Linear(C, C)
v_proj = nn.Linear(C, C)

q_proj1 = nn.Linear(C, D)
k_proj1 = nn.Linear(C, D)
v_proj1 = nn.Linear(C, D)

Q = q_proj(X)
K = k_proj(X)
V = v_proj(X)
Q1 = q_proj1(X)
K1 = k_proj1(X)
V1 = v_proj1(X)

scores = Q @ K.transpose(1, 2)
scores1 = Q1 @ K1.transpose(1, 2)
out = scores @ V
out = scores1 @ V1

print(Q.shape)
print(K.shape)
print(V.shape)
print()
print(Q1.shape)
print(K1.shape)
print(V1.shape)
print("scores.shape =", scores.shape)
print("scores1.shape =", scores1.shape)
print("out.shape =",out.shape)