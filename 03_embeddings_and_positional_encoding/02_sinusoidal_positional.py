import torch

B, T, C = 2, 4, 6
X = torch.zeros(B, T, C)      # 这里用全0是为了方便观察位置编码的效果。真实的transformer中，x通常是token embedding

# 生成位置索引[0, 1, 2, ..., T-1]
pos = torch.arange(T)         # torch.arrange(T)  生成一个从0开始，到T-1结束的一维Tensor

P = torch.zeros(T, C)         # shape为[T, C]的二维Tensor

# 填充前两个位置编码维度        # pos的长度等于P的行数，C的前两个维度用sin和cos填充
P[:, 0] = torch.sin(pos)      # 取 P 中的所有行，但只取第 0 列
P[:, 1] = torch.cos(pos)      # 取 P 中的所有行，但只取第 1 列

Y = X + P

print("P.shape =", P.shape)
print("Y.shape =", Y.shape)



#=========对全部的列引入频率=========
import torch

T, C = 4, 6

pos = torch.arange(T).unsqueeze(1)    # unsqueeze(1)表示在索引1的位置上增加一个长度为1的维度
dim = torch.arange(0, C, 2)

freq = 1/(10000**(dim/C))
angles = pos * freq

P = torch.zeros(T, C)
# 偶数列使用sin
P[:, 0::2] = torch.sin(angles)

# 奇数列使用cos
P[:, 1::2] = torch.cos(angles)

print()
print(P)
print(P.shape)