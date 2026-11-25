#============================================
# mask as a lower triangular matrix (causal mask)
import torch

T = 4

mask = torch.tril(torch.ones(T, T))   # tril=triangle lower 下三角（它会保留下三角，右上角清零，所以会变成元素为1的下三角矩阵）

print(mask)
print()


#=============================================
# masked attention with causal mask

import torch
import torch.nn.functional as F        # 这块是AI写错了，实际上没调用。不过虽然F不等于torch，但是我们可以用F.softmax代替torch.softmax

scores = torch.tensor([
    [2.0, 1.0, 3.0, 0.5],
    [0.2, 1.5, 2.0, 0.7],
    [1.0, 0.3, 2.2, 1.1],
    [0.5, 1.0, 0.2, 2.5]
])

mask = torch.tril(torch.ones(4, 4))   # torch.triu 用来取矩阵的下三角部分，其余位置置为 0。tril 是 lowwer triangular 的缩写。

masked_scores = scores.masked_fill(mask == 0, float("-inf"))

attention_weights = torch.softmax(masked_scores, dim=-1)

print(masked_scores)
print(attention_weights)