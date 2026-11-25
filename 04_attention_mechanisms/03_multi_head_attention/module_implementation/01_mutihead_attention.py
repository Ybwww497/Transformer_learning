import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):                    # nn.Module是pytroch中神经网络参数的基类。之后写的注意力层、全连接网络等，通常都继承它
    def __init__(self, embed_dim, num_heads):           # 注意__init__()函数是两个下划线
        super().__init__()                              # super().__init__() 要先调用，父类初始化完成后，才能正常注册和管理之后定义的神经网络层

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # TODO 1
        # 每个head的维度
        assert embed_dim % num_heads == 0               # 确保embed_dim可以被num_heads整除
        self.head_dim = embed_dim // num_heads          # 注意由于head_dim需要是整数，因此必须用到整除号，将输出的结果转为整数 

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim) 

    def forward(self, x):
        B, T, C = x.shape                               # 当左边变量的个数和右边元素的数量一致时，可以用BTC接住x.shape的三个维度，这在python里叫作序列解包

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # q/k/v 此时：[B, T, C]
        # TODO 2
        # 拆head
        q = q.reshape(B, T, self.num_heads, self.head_dim)             # 注意head_dim必须是整数
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        # TODO 3
        # 调整成[B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        print("q.shape =", q.shape)
        print("k.shape =", k.shape)
        print("v.shape =", v.shape)

        # TODO 4
        # 计算attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # 注意这里是self.head_dim而不是self.embed_dim，原因在于分散到多头以后，attention_score的计算是对于每个head下的维度，也就应该是head_dim，至于embed_dim则是所有head的总维度，这个只有在self_attention的计算中才会使用到

        print("scores.shape =", scores.shape)

        # TODO 5
        # softmax进行归一化attention scores
        attention_weights = torch.softmax(scores, dim=-1)

        print("attention_weights.shape =", attention_weights.shape)

        # TODO 6
        # 计算加权后的V
        out = attention_weights @ v

        print("out.shape =", out.shape)

        # TODO 7
        # 将多头输出先转置再合并(concat heads)

        out = out.transpose(1, 2)
        out = out.reshape(B, T, self.embed_dim)
        out = self.out_proj(out)                     # 这个Linear不是为了改变shape，而是为了让不同的Head拼接后的信息再次进行线性混合

        print("out.shape =", out.shape)

        return out

x = torch.randn(2, 10, 768)

mha = MultiHeadAttention(
    embed_dim=768, 
    num_heads=12
)

result = mha(x)   
print("result.shape =", result.shape)     