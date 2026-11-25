import torch

B, T, V = 2, 4, 1000

logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))

logits_flat = logits.reshape(-1, V)     # -1指的是这个位置的量让计算机自动帮忙计算，一般只有一个-1
labels_flat = labels.reshape(-1)        # logits决定模型倾向预测什么，labels告诉模型正确答案是什么

import torch.nn.functional as F

loss = F.cross_entropy(logits_flat, labels_flat)    # 1.对logits_flat的最后一维计算softmax得到概率  2.根据labels_flat查找正确token的概率   3.计算loss,先计算每个token的交叉熵损失，然后再求平均数
print(loss)                             # loss是每个token的交叉熵损失计算平均数得到的

