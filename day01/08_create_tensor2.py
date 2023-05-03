import torch
import numpy as np


# 正态分布随机数
randn_mat = torch.randn(2,3)
print(randn_mat)
# 均匀分布随机数,范围[0,1]
rand_mat = torch.rand(2,3)
print(rand_mat)
# Int随机,返回[0,10),注意是前闭后开区间
randint_mat = torch.randint(0,10,[3,3])
print(randint_mat)
 
# 二维tensor,可以表示4张mnist图片(图片已fla)
tensor_2d = torch.rand(4,784)
# 三维tensor,可以表示20句话,每句话10个单词,每个单词用onehot来表示[1,100]
tensor_3d = torch.rand(20,10,100)
# 四维tensor,可以表示4张mnist图片,h w都是28,channel为1
tensor_4d = torch.rand(4,1,28,28)
 
# 使用和tensor_4d相同的随机方式和维度定义tensor_4d_2
tensor_4d_2 = torch.rand_like(tensor_4d)
 
# 看tensor_4d有多少元素
print(torch.numel(tensor_4d))
 
# 生成一个元素全是7.0的2*3矩阵
a = torch.full([2,3],7.)
print(a)
# 生成一个元素全是7.0的2维向量
b = torch.full([2],7.)
print(b)
# 生成值为7.0的标量
c = torch.full([],7.)
print(c)

