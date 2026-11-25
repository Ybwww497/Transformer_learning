import torch

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):

    model.eval()

    for _ in range(max_new_tokens):

        # 1. GPT 前向传播
        logits = model(idx)  # [B, T, V]

        # 2. 取最后一个位置的 Logits
        next_logits = logits[:, -1, :]

        # 3. Top-k 筛选
        values, indices = torch.topk(
            next_logits, k=top_k, dim=-1
        )

        # 4. Temperature + Softmax
        probs = torch.softmax(values / temperature, dim=-1)

        # 5. 随机采样一个候选位置
        sampled_idx = torch.multinomial(
            probs, num_samples=1
        )

        # 6. 映射回原始 Token ID
        next_token = torch.gather(
            indices, dim=1, index=sampled_idx
        )

        # 7. 拼接生成的新 Token
        idx = torch.cat([idx, next_token], dim=-1)

    return idx
