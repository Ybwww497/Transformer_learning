import torch

B = 2
T = 4
C = 8

x = torch.randn(B, T, C)

print("x.shape =",x.shape)      
print("x.ndim =",x.ndim)
print("x.dtype =",x.dtype)
print("x.device =",x.device)

print("\n=== index/slice===")
print("x[0].shape =",x[0].shape)
print("x[:, 0, :].shape =",x[:, 0, :].shape)
print("x[:, -1, :].shape =",x[:, -1, :].shape)
print("x[:, -2:, :].shape =",x[:, -2:, :].shape)

H = 2
D = 4
y = x.reshape(B, T, H, D)

print("\n===reshape===")
print("y.shape =",y.shape)

z = y.transpose(1, 2)

print("\n===transpose===")
print("z.shape =",z.shape)

print("\n===dtype===")
a = torch.tensor([1, 2, 3])
print("a.dtype =",a.dtype)
b = a.float()
print("b.dtype =",b.dtype)

print("\n===device===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("selected device =",device)
x = x.to(device)
print("x.device =",x.device)