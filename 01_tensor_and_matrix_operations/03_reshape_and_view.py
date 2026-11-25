import torch

x = torch.arange(12)

print("x =", x)
print("x.shape =", x.shape)

y = x.reshape(3, 4)

print("\ny =", y)
print("y.shape = ", y.shape)

z = x.reshape(2, 2, 3)

print("\nz =", z)
print("z.shape =", z.shape)

#==============Transformer风格的reshape================
import torch

B = 2
T = 4
C = 8

H = 2
D = 4

x = torch.randn(B, T, C)

print("\nbefore:")
print(x.shape)

y = x.reshape(B, T, H, D)

print("\nafter:")
print(y.shape)