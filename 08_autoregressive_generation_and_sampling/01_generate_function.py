import torch

@torch.no_grad()
def generate(model, idx, max_new_tokens):
    """
    model: GPT 模型
    idx: [B, T] 输入 Token IDs
    max_new_tokens: 需要生成的 Tokens 数量
    """
    model.eval()

    for _ in range(max_new_tokens):

        # 1. 前向传播
        logits = model(idx)   # [B, T, V]

        # 2. 只取最后一个位置的Logits
        next_logits = logits[:, -1, :]

        # 3. 使用Greedy Decoding 选择下一个Token
        # 注意保留维度
        next_token = torch.argmax(
            next_logits, dim=-1, keepdim=True
        )

        # 4. 将新 Token 拼接到序列末尾
        idx = torch.cat([idx, next_token], dim=-1)

    return idx