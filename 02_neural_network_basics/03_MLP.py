import torch
import torch.nn as nn

class MLP(nn.Module):              # 两层MLP
    def __init__(self):
        super().__init__()         # 注意__init__是两个下划线

        self.fc1 = nn.Linear(4, 8) # nn.Linear()创建了一个可以被调用的层对象
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)            # self.fc1(x)用nn.Linear()创建的层对象来处理x
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()                      # 按照定义的MLP类，创建一个MLP对象（可以理解为创建一个神经网络结构）

x = torch.randn(32, 4)

pred = model(x)

print(pred.shape)