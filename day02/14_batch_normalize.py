import torch
import torch.nn as nn

x=torch.randint(10,[5,2])+1
x=torch.tensor(x,dtype=torch.float32)

layer=torch.nn.BatchNorm1d(2)

out=layer(x)




# x = torch.tensor([[7, 1],
#                   [2, 1]], dtype=torch.float)
# print(x)
# print(x.shape)   #  x的形状为（3，3）
# m = nn.BatchNorm1d(2)   #  num_features的值必须为形状的最后一数3
# y = m(x)
# print(y)
# # 输出的结果是
# # tensor([[0., 1., 2.],
# #         [3., 4., 5.],
# #         [6., 7., 8.]])
# # torch.Size([3, 3])
# # tensor([[-1.2247, -1.2247, -1.2247],
# #         [ 0.0000,  0.0000,  0.0000],
# #         [ 1.2247,  1.2247,  1.2247]], grad_fn=<NativeBatchNormBackward0>)
# m.running_mean
pass