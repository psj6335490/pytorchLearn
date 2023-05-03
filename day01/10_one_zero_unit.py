import torch
 
# 3*3全一矩阵
a = torch.ones(3,3)
# 生成一个shape和a一样的全一矩阵
a_2 = torch.ones_like(a)
# 3*3零矩阵
b = torch.zeros(3,3)
# 生成一个shape和a一样的零矩阵
b_2 = torch.zeros_like(a)
# 3*3单位矩阵
c = torch.eye(3,3)  # 或torch.eye(3)
# 如果不是方阵,会自动填充0，不会报错
d = torch.eye(3,4)
d_2 = torch.eye(4,3)
print(d)
print(d_2)