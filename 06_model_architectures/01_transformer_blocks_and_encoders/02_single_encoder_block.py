import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, C, H):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim = C,
            num_heads = H,
            batch_first = True
        )

        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)

        self.ffn = nn.Sequential(
            nn.Linear(C, 4*C),
            nn.ReLU(),
            nn.Linear(4*C,C)
        )

    def forward(self, x):
        y = self.norm1(x)

        a,_ = self.attn(y,y,y)
        
        #TODO 1:Attention + Residual
        x = x + a

        #TODO 2:FFN + Residual
        x = x + self.ffn(self.norm2(x))

        return x