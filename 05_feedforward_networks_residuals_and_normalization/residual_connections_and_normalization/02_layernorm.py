import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# 第一步：计算每行平均值
mean = x.mean(dim=-1, keepdim=True)                # dim=-1的原因：LayerNorm只对每个token自身的特征维度C进行归一化，而不把不同的token的特征混在一起计算均值和方差。如果dim=1,此时是batchnorm，批标准化，这时是对一批token的相同维度进行均值和方差计算

# 第二步：计算每行方差                               # keepdim=True, 意为keep dimension, 保留维度, 以x为例：取均值后，保留维度则shape为[2，1], 不保留维度为[2]；此处需要保留是因为x -mean需要维度保持一致，如果维度不一致广播机制无法发挥作用
var = x.var(dim=-1, keepdim=True, unbiased=False)  # unbiased表示无偏，unbiased=False是无偏估计，除以N-1；unbiased=False是有偏估计，除以N；这里LayerNorm用False

# 第三步：标准化（归一化）
eps = 1e-5                # 防止var是0
y = (x - mean) / torch.sqrt(var + eps)             # 注意这里用torch.sqrt()而不是math.sqrt()，var是pytorch张量，里面有多个元素，math.sqrt()只接受单个python的数值，所以用torch.sqrt()

print("mean =", mean)
print("var =", var)
print("y =", y)
print("shape =", y.shape)