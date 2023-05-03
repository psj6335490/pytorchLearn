import torch
from torch.nn import functional as F

x=torch.ones(1)
w=torch.full([1],2.)

mse=F.mse_loss(torch.ones(1),x*w)

w.requires_grad_()
# mse=F.mse_loss(torch.ones(1),x*w)
# #直接计算出梯度
# torch.autograd.grad(mse,[w],retain_graph=True)

mse=F.mse_loss(torch.ones(1),x*w)
#方向传播，把梯度存在对应变量的grad属性上
mse.backward()

pass