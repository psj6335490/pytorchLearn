import torch
 
### 使用concat拼接矩阵
a = torch.rand(3, 4)
b = torch.rand(5, 4)
# 对行拼接，即3行+5行=8行。类似于excel中条目累加
ab_cat = torch.cat([a, b, ], dim=0)
print(ab_cat.size())  # 输出torch.Size([8,4])
 
c = torch.rand(4, 5)
d = torch.rand(4, 6)
# 对列拼接，即5列+6列=11列。类似于excel中不同字段拼接
cd_cat = torch.cat([c, d], dim=1)
print(cd_cat.size())  # 输出torch.Size([4,11])
 
# 在googLenet中对于Inception的拼接，是按channel进行拼接的
res_conv3 = torch.rand(4, 64, 28, 28)
res_conv1 = torch.rand(4, 128, 28, 28)
res = torch.cat([res_conv3, res_conv1], 1)
print(res.size())  # 输出torch.Size([4,192,28,28])
 
### 使用stack组合两个矩阵
aa = torch.rand(32, 8)
bb = torch.rand(32, 8)
# 将两个矩阵组合起来，并且在指定位置创建新维度
# 可以理解为两张图片组成一个batch，而不是两张图片拼在一起
ac_stack = torch.stack([aa, bb], dim=0)
print(ac_stack.size())  # 输出torch.Size([2,32,8])