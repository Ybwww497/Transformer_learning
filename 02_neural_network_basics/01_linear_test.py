import torch

B = 2
T = 10
C_in = 64
C_out = 128

x = torch.randn(2, 10, 64)
w = torch.randn(64 ,128)
b = torch.randn(128)    # b.shape = [128]   但它依然可以加到[2, 10, 128]上，这是因为pytorch会进行broadcasting广播操作，先将b广播成[1，1，128]，再沿着前两个维度复制，相当于变成[2, 10,128],所以每一个样本里的每一个token，都会加上一个128维的bias向量

y = x @ w + b

print("x.shape =", x.shape)
print("w.shape =", w.shape)
print("b.shape =", b.shape)
print("y.shape =", y.shape)



import torch
import torch.nn as nn

x = torch.randn(2, 10, 64)
linear = nn.Linear(64, 128)

print("\nx =", x.shape)
print("weight =", linear.weight.shape)
print("bias =", linear.bias.shape)

y1 = linear(x)
print("y1 =",y1.shape)

y2 = x @ linear.weight.transpose(0, 1) + linear.bias
print(torch.allclose(y1, y2))      # 检查两个Tensor的对应元素是不是足够接近（不是要求每个浮点数完全一摸一样，而是允许非常小的数值误差），在此处是验证：nn.Linear的结果，与手写的x @ W.T + b的结果是否一致，输出True说明复现了nn.Linear的计算

### 自己封装一个Linear函数
import torch

def my_Linear(x, weight, bias):
    y = x @ weight.transpose(0, 1) + bias
    return y

x = torch.randn(2, 10, 64)
weight = torch.randn(128, 64)
bias = torch.randn(128)

y = my_Linear(x, weight, bias)

print()
print(y.shape)