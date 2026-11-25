import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([1.0, 2.0, 3.0])

print(a.dtype)
print(b.dtype)

x = torch.tensor([1.5, 2.6, 3.3])
print(x.dtype)
print(x)
x = x.long()     #把tensor x 转换成torch.int64类型，即转换为64位整数（这里不是四舍五入，而是直接去掉小数部分）
print(x.dtype)
print(x)

#=======================================
#x.float()    torch.float32
#x.bfloat()   torch.bfloat16
#x.half()     torch.float16
#x.long()     torch.int64
#x.int()      torch.int32
#=======================================

#===============dtype转换===============
x = torch.tensor([1, 2, 3])
print()
print(x)
print(x.dtype)
y = x.float()
print(y)
print(y.dtype)

x = torch.randn(2, 3)
print()
print(x.dtype)
y = x.half()
print(y.dtype)
y = x.bfloat16()
print(y.dtype)

#================device==================
import torch

print()
print(torch.cuda.is_available())         #检查电脑是否有cuda以及GPU名称
print(torch.cuda.get_device_name(0))

x = torch.randn(2, 3)
print(x.device)                          #查看tensor所在的设备
x = x.to("cuda")                         #把tensor移动到GPU上
print(x.device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
print(x.device)

#=================实验例子================
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([1.0, 2.0, 3.0])

print("\na.dtype:",a.dtype)
print("b.dtype:",b.dtype)

c = a.float()
print("c.dtype:",c.dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device:",device)
x = torch.randn(2, 3)
print("before:",x.device)
x = x.to(device)
print("after:",x.device)
print("x dtype:",x.dtype)
