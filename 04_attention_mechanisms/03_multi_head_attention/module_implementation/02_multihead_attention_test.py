import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)   

        attention_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)        
        attention_weights = torch.softmax(attention_scores, dim=-1)

        output = attention_weights @ v
        output = output.transpose(1, 2).reshape(B, T, C)
        output = self.out_proj(output)

        return output

x = torch.randn(2, 10, 768)

mda = MultiHeadAttention(embed_dim=768, num_heads=8)

output = mda(x)
print("output.shape =", output.shape)