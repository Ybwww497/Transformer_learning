#===========================================
# 测试不进行scaled的softmax后果

import torch
import math

scores = torch.tensor([1.0, 2.0, 3.0])

# TODO 1: 计算scores的softmax
softmax1 = torch.softmax(scores, dim=-1)

scores_big = scores * 10

# TODO 2: 计算scores_big的softmax
softmax2 = torch.softmax(scores_big, dim=-1)

print("softmax1=", softmax1)
print("softmax2=", softmax2)




#===========================================
# 验证scaling
import torch
import math

scores = torch.tensor([8.0, 16.0, 24.0])
d_k = 64

# 不进行saling
attention_score1 = torch.softmax(scores, dim=-1)

# TODO: 除以sqrt(d_k)
scaled_scores = scores / math.sqrt(d_k)    ### 这里不能直接用sqrt(),因为这个方法没有被导入或定义，可以用math.sqrt(d_k)或在开头前面from math import sqrt    

# 再进行softmax
attention_score2 = torch.softmax(scaled_scores, dim=-1)

print("原始scores:", scores)
print("scaled_scores:",scaled_scores)

print("不scaling:",attention_score1)
print("scaling后:",attention_score2)

#=============================================
# 把scaling放回attention中
import torch
import math
from math import sqrt

Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)

d_k = Q.shape[-1]

# Q和K做矩阵乘法
scores = Q @ K.transpose(-2, -1)

# scaling
scaled_scores = scores / sqrt(d_k)

# softmax
attention_weights = torch.softmax(scaled_scores, dim=-1)

# attention_weights 和 V相乘
output = attention_weights @ V

print("\nQ.shape:", Q.shape)
print("score_shape:", scores.shape)
print("attention_weights_shape:", attention_weights.shape)
print("output_shape:", output.shape)