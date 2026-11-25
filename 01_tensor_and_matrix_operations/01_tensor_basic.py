import torch     # pytorch 里对tensor进行操作的库

# 创建0维tensor
a = torch.tensor(5)  # 创建一个tensor

print("a =",a) 
print("a.shape = ",a.shape)       # 打印tensor的形状          
print("a.ndim = ",a.ndim)         # 打印tensor的维度数        

# 创建1维tensor
b = torch.tensor([1, 2, 3, 4])

print("\nb = ",b)             
print("b.shape = ",b.shape)       # shape = [4]
print("b.ndim = ",b.ndim)         # 1维tensor,长度是4

# 创建2维tensor
c = torch.tensor([[1, 2, 3],[4, 5, 6]])

print("\nc = ",c)
print("c.shape = ",c.shape)       # shape = [2, 3]
print("c.ndim = ",c.ndim)         # 2维tensor,形状是(2, 3),共有6个元素

# 创建3维随机tensor
x = torch.randn(2, 3, 4)

print("\nx = ",x)
print("x.shape = ",x.shape)       # shape = [2, 3, 4]       
print("x.ndim = ",x.ndim)         # 3维tensor,形状是(2, 3, 4),共有24个元素

x = torch.tensor([[10, 20, 30],[40, 50, 60]])
print("\nx[0] = ",x[0])             # 获取tensor的某一行
print("x[0][1] = ",x[0][1])       # 获取tensor的某一个具体数字，第几行第几列

x = torch.tensor([[10, 20, 30],[40, 50, 60]])

# ":" 表示全选
print("\nx[:,0] = ",x[:,0])        # 输出 tensor([10, 40])
print("x[:,1] = ",x[:,1])        # 输出 tensor([20, 50])
print("x[0,:] = ",x[0,:])        # 输出 tensor([10, 20, 30])
print("x[1,:] = ",x[1,:])        # 输出 tensor([40, 50, 60])


# 三维tensor例题
x = torch.randn(2, 4, 8)

print("\n整个tensor:", x.shape)
print("x[0]:", x[0].shape)
print("x[0][0]:",x[0][0].shape)

# 三个常用的tensor创建方法

## 全是0
x = torch.zeros(2, 3)

print("\nx = ",x)

## 全是1
x = torch.ones(2, 3)

print("x = ",x)

## 随机数，深度学习里randn()函数非常常见
x = torch.randn(2, 3)

print("x = ",x)


