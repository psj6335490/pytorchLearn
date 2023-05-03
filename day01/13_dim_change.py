import torch
 
a = torch.rand(4, 1, 28, 28)
# opencv中图片的格式也是hwc。但在pytorch中为CHW
a_1 = a.view(4, 784)
print(a_1.size())
a_2 = a.view(4, 1, 28, 28)
print(a_2.size())
a_3 = a.view(4 * 1 * 28, 28)
print(a_3.size())
# 尽量不要这样转，因为乱转维度可能破坏数据的几何特性
a_4 = a.view(4, 28, 28, 1)
print(a_4.size())