import torch
 
src = torch.rand(4, 32, 14, 14)
b = torch.rand(1, 32, 1, 1)
 
### 使用expand来扩展维度
### 注意，被扩展的维度只能是1-->n，而不能是m-->n。数据会自动复制
# 将c扩展为torch.Size([4,32,14,14])
c = b.expand(4, 32, 14, 14)
# 将c扩展为和src一样的维度
d = b.expand_as(src)
print(c.size())
# print(c)
print(d.size())
# print(d)
 
# 只指定需要扩展的维度，其他维度不动可以填-1
e = b.expand(4, -1, -1, -1)
print(e.size())  # 输出torch.Size([4,32,1,1])
 
##====================================##
## 使用repeat来扩展维度
# repeat的参数不是代表扩展后的维度，而是分别需要复制多少次
f = b.repeat(4, 1, 14, 14)
print(f.size())  # 扩展后的维度为torch.Size([4,32,14,14])