import torch

B = 2
T = 4
C = 8

Q = torch.randn([2, 4, 8])
K = torch.randn([2, 4, 8])
V = torch.randn([2, 4, 8])

K_T = K.transpose(1, 2)
scores = Q @ K_T
out = scores @ V

print("Q.shape =",Q.shape)
print("K_T.shape =",K_T.shape)
print("scores.shape =",scores.shape)
print("out.shape =",out.shape)
