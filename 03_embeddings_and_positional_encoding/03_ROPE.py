import torch

def rotate(v, angle):
    x, y = v[0], v[1]

    x_new = x * torch.cos(angle) - y * torch.sin(angle)
    y_new = x * torch.sin(angle) + y * torch.cos(angle)

    return torch.stack([x_new, y_new])      # torch.stack把两个标量重新组合成一个一维向量

q = torch.tensor([1.0, 0.0])
k = torch.tensor([1.0, 0.0])

# 每增加一个位置，旋转20度
omega = torch.deg2rad(torch.tensor(20.0))   # torch.deg2rad 角度转弧度，rope的输入需要是弧度

for m, n in [(2, 5), (10, 13)]:             # 一个列表，包含两个元组，每个元组包含两个元素。python会自动将元组中的两个值分别赋给m和n
    q_rot = rotate(q, m * omega)
    k_rot = rotate(k, n * omega)

    score = torch.dot(q_rot, k_rot)         # torch.dot 求两个向量的点积
    print(f"位置({m}, {n}): {score.item():.4f}")    # f"..." 格式化字符串，使得可以通过{}插入变量
                                                    # score.item() 提取tensor中的数值   :.4f 保留小数点后四位