import torch
import numpy as np
 
# 建议使用torch.tensor()来直接赋值
a = torch.tensor([1., 2., 3.])  # 直接赋值（建议）
# 不建议用FloatTensor来直接赋值，避免混淆
a_2 = torch.FloatTensor([1.,2.,3.]) # 也可以用FloatTensor赋值
 
# 建议使用FloatTensor传入shape来定义数据结构
b = torch.FloatTensor(1)  # 参数表示shape，这里是2个元素的向量，值未初始化，可能很大或很小
c = torch.FloatTensor(3, 2)  # 这里表示维度为[3,2]的矩阵，值未初始化，可能很大或很小
 
d = torch.ones(3, 3)  # 定义维度为[3,3]的全1矩阵
 
# 同numpy来转换数据
e_np = np.ones((3, 3))  # 定义numpy的全1 ndarray
e = torch.from_numpy(e_np)  # 使用numpy转换到tensor
 
f=torch.tensor(2)

print('a: ', a)
print('b: ', b)
print('c: ', c)
print('d: ', d)
print('e: ', e)
print('f: ', f.shape)
