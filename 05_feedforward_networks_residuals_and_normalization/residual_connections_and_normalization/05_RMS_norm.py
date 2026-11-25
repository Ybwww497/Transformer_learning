import torch

x = torch.tensor([
    [
        [2., 4., 6.],
        [1., 2., 3.]
    ]
])

print(x.shape)

def rms_norm(x, eps=1e-6):
    # 1. 在最后一维计算平方均值    x.pow(2)：每个元素平方   .mean(dim=-1)：对最后一个维度求平均  keepdim = True 表示保留维度，不会因为求平均得到一个数字就消除最后一维度   
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    
    # 2. 计算RMS(加eps)
    rms = torch.sqrt(mean_square + eps)

    # 3. 将x除以PMS
    return x / rms

x = torch.randn(2, 5, 64)
y = rms_norm(x)

print(y.shape)