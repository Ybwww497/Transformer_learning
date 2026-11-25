#============x.shape的验证============
import torch

B, T, C = 2, 4, 8

x = torch.randn(B, T, C)

# 假设这是Attention的输出
attention_output = torch.randn(B, T, C)

# 假设这是ffn的输出
ffn_output = torch.randn(B, T, C)

# TODO 1: 实现第一次残差连接
x = x + attention_output

# TODO 2: 实现第二次残差连接：
x = x + ffn_output

print("x.shape:", x.shape)


#============ffn模块手写============
import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, C):
        super().__init__()
        
        self.fc1 = nn.Linear(C, 4*C)
        self.fc2 = nn.Linear(4*C, C)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x

B, T, C = 2, 4, 8
x = torch.randn(B, T, C) 
ffn = FFN(C)
y = ffn(x)
print()
print("y.shape:", y.shape)

#============引入Multi-Head_Attention============
class TransformerBlock(nn.Module):
    def __init__(self, C, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(C)
        self.attention = nn.MultiheadAttention(
            embed_dim = C,
            num_heads = num_heads,
            batch_first = True           # batch_first=True 表示：输入和输出张量的第一个维度是 batch（批次大小），即将把张量的格式设置为(B, T, C), 否则是(T, B, C)
        )

        self.norm2 = nn.LayerNorm(C)
        self.ffn = FFN(C)

    def forward(self, x):
        B, T, C = x.shape

        # Causal Mask:防止看到未来Token
        mask = torch.triu(                                           # torch.triu 用来取矩阵的上三角部分，其余位置置为 0。triu 是 upper triangular 的缩写。
            torch.ones(T, T, device=x.device, dtype=torch.bool),     # 创建一个T*T矩阵，元素全为1，让矩阵和x放在同一个设备上，元素使用布尔类型(所以这里的1是True)
            diagonal=1                                               # 保留严格位于主对角线上方的部分（diagonal是对角线的意思）
        )                                                            # 后续会把元素为1的位置赋成-inf，经过softmax后就会变成0了，然后权重为0就不考虑了

        # 第一部分：Attention + Residual
        a = self.norm1(x)

        attention_output, _ = self.attention(          # _ 表示这个值返回后不用    
            query=a,
            key=a,
            value=a,
            attn_mask=mask,
            need_weights=False                         # need_weights=False 表示：只需要注意力计算的结果，不需要返回注意力权重矩阵，TRUE时表示返回torch.softmax(Q@K.transpose(1, 2)),即返回注意力权重矩阵
        )

        x = x + attention_output

        # 第二部分：FFN + Residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x

B, T, C = 2, 4, 8
x = torch.randn(B, T, C)
block = TransformerBlock(C=8, num_heads=2)
y = block(x)
print("输出Shape:", y.shape)
