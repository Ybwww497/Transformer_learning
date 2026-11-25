import torch
import math

B, T, C = 2, 5, 8

Q = torch.randn(B, T, C)
K = torch.randn(B, T, C)
V = torch.randn(B, T, C)

scores = Q @ K.transpose(1, 2)
scores = scores / math.sqrt(C)
weights = torch.softmax(scores, dim = -1)
out = weights @ V

print(scores.shape)
print(weights.shape)
print(out.shape)




import torch
import math

B, T, C = 2, 5, 8

X = torch.randn(B, T, C)

W_q = torch.nn.Linear(C, C, bias=False)
W_k = torch.nn.Linear(C, C, bias=False)
W_v = torch.nn.Linear(C, C, bias=False)

Q = W_q(X)
K = W_k(X)
V = W_v(X)

scores = Q @ K.transpose(1, 2)
scores = scores / math.sqrt(C)
weights = torch.softmax(scores, dim=-1)
out = weights @ V

print("\nQ:", Q.shape)
print("scores:", scores.shape)
print("weights:", weights.shape)
print("out:", out.shape)



import torch
import torch.nn as nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.C = C

        self.W_q = nn.Linear(C, C) 
        self.W_k = nn.Linear(C, C)
        self.W_v = nn.Linear(C, C)

    def forward(self, X):

        Q = self.W_q(X) 
        K = self.W_k(X)
        V = self.W_v(X)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.C)   # 最好不要直接用外面的全局变量C,用self.C
        weights = torch.softmax(scores, dim=-1)
        out = weights @ V

        return out

B = 2
T = 5
C = 8
X = torch.randn(B, T, C)
singleheadattention = SingleHeadAttention(C)
OUT = singleheadattention(X)                  # 虽然能运行，但 PyTorch 里通常不要直接调 .forward()    因为 nn.Module 的 __call__ 会自动帮你调用 forward()，同时还会处理 hook 等机制
print("\nout =", OUT.shape) 
