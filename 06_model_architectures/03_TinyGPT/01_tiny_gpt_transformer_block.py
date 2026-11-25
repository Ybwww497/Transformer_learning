import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3*d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        # 一次线性变换产生Q、K、V
        qkv = self.qkv(x)

        # TODO 1: 拆分Q、K、V
        q, k, v = qkv[:,:,:C], qkv[:,:,C:2*C],qkv[:,:,2*C:3*C]
        # 或者使用：q, k, v = qkv.chunk(3, dim=-1)

        # TODO 2: 把Q拆分成多个Head
        q = q.view(B, T, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)

        # TODO 3：把K拆分成多个Head
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # TODO 4: 把V拆成多个Head
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # TODO 5：计算Attention Scores
        scores = q @ k.transpose(-1, -2)

        # TODO 6: 进行Scaling
        scores = scores / math.sqrt(self.head_dim)

        mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        # TODO 7:将未来Token的Scores设为-inf
        scores = scores.masked_fill(mask==False, float("-inf"))

        # TODO 8:计算Softmax
        attn = torch.softmax(scores, dim=-1)

        # TODO 9:计算Attention输出
        out = attn @ v

        # TODO 10:交换维度
        out = out.transpose(1, 2)

        # TODO 11:合并多个Head[B, T, H, D] -> [B, T, C]
        out = out.reshape(B, T, C)

        # TODO 12:输出投影
        out = self.out_proj(out)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn =  CausalSelfAttention(d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )

    def forward(self, x):
        #TODO 1:Norm + Attention + Residual
        x = x + self.attn(self.norm1(x))
        
        #TODO 2:Norm + FFN + Residual
        x = x + self.ffn(self.norm2(x))

        return x
    
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context_len, d_model):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)

        # self.blocks = nn.Identity()

        self.lm_head = nn.Linear(d_model, vocab_size)

        # 深度网络，创建两层block
        # TODO 1:创建两个Transformer Block     创建两个独立的Transformer Block实例，并将它们存放在self.blocks这个容器里
        self.blocks = nn.ModuleList([         # nn.ModuleList负责存放和注册模块，并不会定义如何把输入传给这些模块，因此不能直接用self.blocks(x)
            TransformerBlock(d_model),        # 如果是nn.Sequential来定义这两个模块，就可以直接调用self.blocks(x)，nn.Sequential会自动执行两个Block
            TransformerBlock(d_model),        # ModuleList负责保存模块，Sequential负责保存并按顺序执行模块
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        x = tok + pos
        # TODO 2: 依次经过两个Transformer Block     需要通过循环遍历这个容器里的两个模块
        for block in self.blocks:      
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits


model = TinyGPT(vocab_size=1000, context_len=128, d_model=64)

idx = torch.randint(0, 1000, (2, 10))

logits = model(idx)

print("logits.shape =", logits.shape)

assert logits.shape == (2, 10, 1000)   # 形状符合[2, 10, 1000]时，程序继续执行；不符合时，会抛出 AssertionError，帮助你发现数据维度错误