import torch
import torch.nn as nn
import math

B = 2
T = 10
C = 768
H = 12

x = torch.randn(B, T, C)

q_proj = nn.Linear(C, C)
k_proj = nn.Linear(C, C)
v_proj = nn.Linear(C, C)

Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)

D = C // H

Q = Q.reshape(B, T, H, D)
K = K.reshape(B, T, H, D)
V = V.reshape(B, T, H, D)

Q = Q.transpose(1, 2)
K = K.transpose(1, 2)
V = V.transpose(1, 2)

attention_scores = Q @ K.transpose(-2, -1) 

print("Q.shape:", Q.shape)
print("K.shape:", K.shape)
print("V.shape:", V.shape)
print("attention_scores.shape:", attention_scores.shape)

attention_scores = attention_scores / math.sqrt(D)

mask = torch.tril(torch.ones(T, T))
attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
attention_weights = torch.softmax(attention_scores, dim=-1)

out = attention_weights @ V
out = out.transpose(1, 2).reshape(B, T, C)

print("out.shape:", out.shape)