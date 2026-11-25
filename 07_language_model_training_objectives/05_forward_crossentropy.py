import torch
import torch.nn as nn
import torch.nn.functional as F

V = 100
C = 16

# 暂时代替TinyGPT                         这个模型有两组可训练权重，embedding权重的shape是[100, 16], Linear的权重的shape是[100, 16]
model = nn.Sequential(
    nn.Embedding(V, C),                  # nn.Embedding(V, C) 是 PyTorch 的嵌入层，把整数编号转换成长度为 C 的可训练向量。  按着x最后一维里token的编号，找到对应token的维度
    nn.Linear(C, V)
)

tokens = torch.tensor([
    [10, 20, 30, 40, 50],
    [11, 21, 31, 41, 51]
])

# Shift Labels
x = tokens[:,:-1]
y = tokens[:, 1:]

# Forward
logits = model(x)

# Flatten(使变平)
logits_flat = logits.reshape(-1, V)
labels_flat = y.reshape(-1)

# CrossEntropy
loss = F.cross_entropy(logits_flat, labels_flat)

print("logits.shape =", logits.shape)
print("loss =", loss.item())            # loss.item()把只有一个元素的 PyTorch 张量，转换成普通 Python 数值，方便打印或记录

# 反向传播                               当执行loss.backward()时，PyTorch 会通过自动求导（Autograd）计算梯度，并将结果保存到对应参数的 .grad 属性中
loss.backward()                         # loss.backward() 的作用是计算 Loss 对模型各个可训练参数的梯度，而不是直接修改模型参数

print(model[1].weight.grad.shape)

# 创建优化器
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)             # SGD：随机梯度下降优化器     lr=0.01：学习率     

# 更新模型参数
optimizer.step()                                                    # optimizer.step()：使用之前 backward() 计算的梯度更新模型参数





#===================================在循环中多次进行梯度计算与权重更新=================================================

# for step in range(10):
#     optimizer.zero_grad()                                           # 清空旧梯度
    
#     logits = model(x)                                               # 前向传播
    
#     loss = F.cross_entropy(                                         # 计算损失
#         logits.reshape(-1, V),
#         y.reshape(-1)
#     )

#     loss.backward()                                                 # 计算梯度

#     # 更新模型参数
#     optimizer.step()                                                # optimizer.step()：使用之前 backward() 计算的梯度更新模型参数

#     print(f"step={step},loss={loss.item():4f}")