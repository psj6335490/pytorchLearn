import torch
 
### 使用split拆分矩阵
a = torch.rand(2, 32, 8)
# 平均拆分
a1, a2 = a.split(1, dim=0)
print(a1.size())  # torch.Size([1,32,8])
 
b = torch.rand(7, 32, 8)
# 按个数拆分
b1, b2, b3 = b.split([3, 3, 1], dim=0)
print(b1.size())  # torch.Size([3,32,8])
 
### 使用chunk拆分矩阵
c = torch.rand(8, 32, 8)
# 将c拆分在dim=0上拆分为两半
c1, c2 = c.chunk(2, dim=0)
print(c1.size())
# 拆分为4份
c3, c4, c5, c6 = c.chunk(4, dim=0)
print(c3.size())
# 拆分为3份，3+3+2
c7, c8, c9 = c.chunk(3, dim=0)
print(c7.size(), c8.size(), c9.size())

 
 
x = torch.rand(4,8,6)
y = torch.split(x,2,dim=0) #按照4这个维度去分，每大块包含2个小块
for i in y :
    print(i.size())
 
 
 
y = torch.split(x,3,dim=0)#按照4这个维度去分，每大块包含3个小块
for i in y:
    print(i.size())
 
 
 
x = torch.rand(4,8,6)
y = torch.split(x,[2,3,3],dim=1)
for i in y:
    print(i.size())
 
 
 
y = torch.split(x,[2,1,3],dim=1) #2+1+3 等于6 != 8 ,报错
for i in y:
    print(i.size())
 
 