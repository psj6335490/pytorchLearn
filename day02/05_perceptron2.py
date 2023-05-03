import torch
from torch.nn import functional as F

x=torch.randn(1,10)
w=torch.randn(2,10,requires_grad=True)

w.retain_grad=True
mu=x@w.t()

o=torch.sigmoid(mu)
print(o.shape)
loss=F.mse_loss(torch.ones(1,1),o)
loss.backward()

print(w.grad)

loss=F.mse_loss(torch.ones(1,2),o)
loss.backward()
print(w.grad)