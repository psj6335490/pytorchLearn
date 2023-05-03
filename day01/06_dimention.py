import torch

a = torch.randn(2, 3)

# b是一个dim为0的标量（就是一个数）
b = torch.tensor(2.2)

# 查看shape
print(a.shape)  # 返回torch.Size([2,3])
print(b.shape)  # 返回torch.Size([])
print(len(a.shape))  # 返回2
print(len(b.shape))  # 返回0,表示dim为0
# size()和shape是一样的，size是成员函数，shape是成员属性
print(a.size())  # 返回torch.Size([2,3])
print(a.size(0))  # 返回2
print(a.size(1))  # 返回3
print(b.size())  # 返回torch.Size([])
# 返回a的维度，返回2，表示2D矩阵
print(a.dim())
c = torch.randn(1)
print(c.shape)