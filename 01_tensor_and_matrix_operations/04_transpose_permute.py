import torch

B = 2
T = 4
C = 8

H = 2
D = 4

x = torch.randn(B, T, C)

print("原始：")
print(x.shape)

x = x.reshape(B, T, H, D)
print("\nreshape后：")
print(x.shape)

x = x.transpose(1,2)
print("\ntranspose后：")
print(x.shape)