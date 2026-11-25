import torch

w = torch.tensor(2.0, requires_grad=True)    # 创建数值为2.0的tensor,requires_grad=True表示“记录计算过程，因为之后需要计算Loss对w的梯度”（即保留前向计算过程） 
loss = (w - 5) ** 2                          
print("loss =",loss.item())                  # loss此时是一个Tensor，.item()表示把只有一个数字的Tensor取成普通python数字
loss.backward()                              # 自动将L对w的偏导数求出来，并保存在w.grad里
print("w.grad =",w.grad)      