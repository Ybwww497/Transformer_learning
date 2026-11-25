import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads==0
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim)
        K = K.reshape(B, T, self.num_heads, self.head_dim)
        V = V.reshape(B, T, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attention_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T))    # mask = torch.tril(torch.ones(T, T, device=x.device))  与x所在的设备保持一致
        attention_scores = attention_scores.masked_fill(mask ==0, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = attention_weights @ V
        output = output.transpose(1, 2).reshape(B, T, self.embed_dim)
        out = self.out_proj(output)

        return out

x = torch.randn(2, 10, 768)
mha = MultiHeadAttention(768, 8)
out = mha(x)
print("out.shape =", out.shape)

