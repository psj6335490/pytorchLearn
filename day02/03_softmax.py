import torch
from torch.nn import functional as F

a=torch.rand(3)
a.requires_grad_()
a
print(a)

p=F.softmax(a,dim=0)

torch.autograd.grad(p[1],[a])
torch.autograd.grad(p[2],[a])

pass
