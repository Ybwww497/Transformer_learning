import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, C, H):
        super().__init__()

        self.norm1 = nn.LayerNorm(C)

        self.attn = nn.MultiheadAttention(      # 可以理解为：nn.MultiheadAttention也是一个类，前面的参数是在创建时需要赋予的参数，后面的参数是在具体调用时使用的参数
            embed_dim=C,
            num_heads=H,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(C)

        self.ffn = nn.Sequential(
            nn.Linear(C, 4*C),
            nn.GELU(),
            nn.Linear(4*C, C)
        )

    def forward(self, x):
        B, T, C = x.shape

        # TODO 1:构建Causal Mask
        # 要求Shape为[T, T]
        # True表示禁止关注
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1              # diagonal=1,只屏蔽严格位于主对角线上方的未来位置
        )

        # Attention
        z = self.norm1(x)
        attn_out, _ = self.attn(     # 可以理解为：nn.MultiheadAttention也是一个类，前面的参数是在创建时需要赋予的参数，后面的参数是在具体调用时使用的参数
            z, z, z,                 # query=z, key=z, value=z
            attn_mask=mask,          
            need_weights=False       # need_weights=False 表示：只需要注意力计算的结果，不需要返回注意力权重矩阵，TRUE时表示返回torch.softmax(Q@K.transpose(1, 2)),即返回注意力权重矩阵
        )

        # TODO 2:第一次残差连接
        x = x + attn_out

        # TODO 3:第二次残差连接
        x = x + self.ffn(self.norm2(x))    

        return x

x = torch.randn(2, 4, 8)

model = DecoderBlock(C=8, H=2)
y = model(x)

print("y.shape =", y.shape)

################################# 修改未来token,观察之前token的输出是否受到影响

model.eval()     # 调用模型对象的 eval() 方法，将模型切换到评估模式。它会影响某些层的行为，比如 Dropout 在评估模式下不再随机丢弃元素，方便我们比较两次输出。注意：eval() 本身不会关闭梯度计算。 关闭梯度由下一句完成。

with torch.no_grad():       # with 是 Python 的一种语法，用来让一段代码在指定的环境下运行。 在下面缩进的代码中，不记录用于计算梯度的信息。通常用在只需要计算结果、不需要反向传播的场景，可以减少内存开销  
    # 原始输入，调用模型对象，会通过 __call__ 机制执行 forward()
    y1 = model(x)

    # 复制输入，只修改最后一个 Token
    x2 = x.clone()                                # clone() 会复制一个内容相同、拥有独立数据存储的张量。
    x2[:, 3, :] = torch.randn_like(x2[:, 3, :])   # randn_like创建一个形状、数据类型和设备与给定张量一致的新张量，里面填入标准正态分布的随机数

    # 再次前向传播
    y2 = model(x2)

    # 比较前面三个 Token 的输出
    print("前3个Token是否相同:")
    print(torch.allclose(y1[:, :3, :], y2[:, :3, :], atol=1e-6))       # :3相当于 0：3，即0、1、2
                                                                       # atol是绝对误差
    # 比较最后一个 Token 的输出
    print("最后1个Token是否相同:")
    print(torch.allclose(y1[:, 3, :], y2[:, 3, :], atol=1e-6)) 