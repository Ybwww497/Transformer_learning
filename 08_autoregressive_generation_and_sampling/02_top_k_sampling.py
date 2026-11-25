import torch

logits = torch.tensor([
    [5.0, 2.0, 4.0, 1.0, 3.0]
])

temperature = 0.8
top_k = 3

# 1. 选出最大的k个Logits和对应的Token ID
values, indices = torch.topk(
    logits, k=top_k, dim=-1
)

# 2. Temperature缩放
scaled_logits = values / temperature

# 3. Softmax转换成概率
probs = torch.softmax(scaled_logits, dim=-1)

# 4. 根据概率采样一个候选位置
sampled_idx = torch.multinomial(
    probs, num_samples=1
)

# 5. 把候选位置映射回原始Token ID
next_token = torch.gather(
    indices, dim=1, index=sampled_idx           # sampled_idx 是 Top-k 候选列表中的位置，而 next_token 才是原始词表中的 Token ID
)

print(next_token)