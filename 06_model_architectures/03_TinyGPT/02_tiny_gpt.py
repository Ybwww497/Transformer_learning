import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context_len, d_model):
        super().__init__()
        
        # nn.Embedding 生成可训练的向量查找表
        self.token_emb = nn.Embedding(vocab_size, d_model)             # token_emb：形状为 [1000, 64]，根据 token 编号取出对应的向量
        self.pos_emb = nn.Embedding(context_len, d_model)              # pos_emb：形状为 [128, 64]，根据位置编号取出对应的向量

        # 暂时用Identity占位，后面替换成Transformer Blocks               # nn.Identity() 是一个原样返回输入的层：输入什么，就输出什么，不做任何计算变换
        self.blocks = nn.Identity()
        
        self.lm_head = nn.Linear(d_model, vocab_size)                  # 将向量维度从d_model转为vocab_size词表大小的线性变换（lm_head可以意译为：语言模型的输出层；head在深度学习里指接在模型主干后面，负责某个具体任务的输出层）

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))        # 等价于两句话   positions = torch.arange(10, device=idx.device)
                                                                      #               pos = self.pos_emb(positions)    取出前10行对应的位置向量
        # TODO 1:组合Token Embedding 和 Position Embedding
        x = tok + pos

        x = self.blocks(x)

        # TODO 2:输出Logits
        logits = self.lm_head(x)

        return logits

model = TinyGPT(vocab_size=1000, context_len=128, d_model=64)    # vocab_size=1000:词表有1000个token    context_len=128：支持的位置为128个token   d_model=64：每个 token 用一个 64 维向量表示

# torch.randn生成随机浮点数，torch.randint生成随机整数
idx = torch.randint(0, 1000, (2, 10))        # 生成shape为[2，10]的整数张量，B为2，T为10，整数的范围为[0, 1000),表示token_id                        

logits = model(idx) 

print(logits.shape)