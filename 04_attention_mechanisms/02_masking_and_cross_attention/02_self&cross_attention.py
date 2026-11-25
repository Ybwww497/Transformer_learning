import torch
import torch.nn as nn

B = 2
Tq = 3
Tk = 5
C = 8

x_query = torch.randn(B, Tq, C)
x_context = torch.randn(B, Tk, C)

Wq = nn.Linear(C, C)
Wk = nn.Linear(C, C)
Wv = nn.Linear(C, C)

Q = Wq(x_query)
K = Wk(x_context)
V = Wv(x_context)

scores = Q @ K.transpose(-2, -1)

print("Q.shape =", Q.shape)
print("K.shape =", K.shape)
print("V.shape =", V.shape)
print("scores.shape =", scores.shape)