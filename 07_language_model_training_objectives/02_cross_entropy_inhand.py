import torch
import torch.nn.functional as F

logits = torch.tensor([
    [1.0, 5.0, 2.0],
    [4.0, 1.0, 2.0]
])

labels = torch.tensor([1, 2])

# 取每行分数最高的Token ID
pred = logits.argmax(dim=-1)

# 计算平均交叉熵损失
loss = F.cross_entropy(logits, labels)

print("pred =", pred)
print("labels =", labels)
print("loss =", loss)