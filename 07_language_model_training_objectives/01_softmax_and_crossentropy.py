import torch
import torch.nn as nn

logits = torch.tensor([[1.0, 0.5, 3.0]])
target = torch.tensor([2])        # 这里指的是token的索引，也就是3.0。注意target的数据类型是一个标量tensor

loss_fn = nn.CrossEntropyLoss()   # 注意CrossEntropyLoss接收的是logits，不是Softmax后的概率，因为CrossEntropyLoss内部已经完成与Softmax/log probability相关的处理
loss = loss_fn(logits, target)    # 计算交叉熵损失需要传一个Logits和正确token索引的参数

print(loss)




import torch
import torch.nn as nn

logits = torch.tensor([[1.0, 0.5, 3.0],[2.0, 4.0 ,1.0]])
target = torch.tensor([2, 1])

# 任务一：对Logits做softmax         注意torch.softmax是可以直接调用的函数，nn.Softmax是需要先进行实例化的神经网络模块
probs = torch.softmax(logits, dim=-1)
# 或者用
# softmax = nn.Softmax(dim=-1)     把这个神经网络模块类进行实例化，再调用对象softmax
# probs = softmax(logits)

# 任务二：创建CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# 任务三：计算Loss
loss = loss_fn(logits, target)

print("\nprobs =", probs)
print("每行概率和 =", probs.sum(dim=-1))
print("loss =", loss)



import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

logist1 = torch.tensor([[1.0, 0.5, 1.0]])
target = torch.tensor([2])
loss1 = loss_fn(logist1, target)
print("\nloss1 =", loss1)

logist2 = torch.tensor([[1.0, 0.5, 3.0]])
loss2 = loss_fn(logist2, target)
print("loss2 =", loss2)