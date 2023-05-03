import torch
import numpy as np
 
a = torch.rand(4, 3, 32, 32)
# 基本索引(和numpy类似)
print(a[2][1][15][15])
print(a[2, 1, 15, 15])
 
# 切片索引(和numpy类似)
print(a[:2, :-1, 3:6, 7:9].size())
print(a[:1, :, :, :].size())
 
# 带步长的切片索引(和numpy类似)
print(a[:, :2, :18:2, ::3].shape)
 
# 指定某一个维度截取,例如取0,1和第3张图片
print(a.index_select(0, torch.tensor([0, 1, 3])).size())
# 取所有图片，但只取0和2个channel
print(a.index_select(1, torch.tensor([0, 2])).size())
# 取图片的上半部分
print(a.index_select(2, torch.arange(0, 14)).size())
# 取图片的右半部分
print(a.index_select(3, torch.arange(14, 28)).size())
 
# 使用...来方便取值
print(a[0, ...].size())
print(a[:, :2, ...].size())
print(a[..., :13, :].size())
 
# 使用mask来取值
b = torch.randn(5, 5)
# 大于0.5的位置为1，小于0.5的位置为0
mask = b.ge(0.5)
print(mask.type())  # type为ByteTensor
# 得到的b_seleted是一个向量，和b的维度没有关系
b_seleted = torch.masked_select(b, mask)
print(b_seleted.size())  # 输出torch.Size(7)，根据b中数据大于0.5的元素个数
 
# 对flatten以后的数据按index取值（不常用）
token = torch.take(b, torch.tensor([2, 6, 13, 22, 24]))
print(token.size())  # 输出torch.Size(5)