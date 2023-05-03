import torch
 
## 添加维度
src1 = torch.rand(4,1,28,28)
 
# 在size的index=0的位置插入一个维度,比如理解为batch,每个batch有4张图片
b = src1.unsqueeze(0)
print(b.size())  # 输出torch.Size([1, 4, 1, 28, 28])
# 在size的最后一个位置插入一个维度
c = src1.unsqueeze(-1)
print(c.size())  # 输出torch.Size([4, 1, 28, 28, 1])
 
##======================================##
## 删除维度
src2 = torch.rand(1,32,3,1,4)
 
# 删除所有可以删除的维度
d = src2.squeeze()
print(d.size())
# 删除第一个维度
e = src2.squeeze(0)
print(e.size())
# 删除最后一个维度
f = src2.squeeze(-2)
print(f.size())