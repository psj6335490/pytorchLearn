import torch

from torch.nn import functional as F

a=torch.linspace(-100,100,10)
print(a)

c=1/(1+torch.exp(-a))
print(c)

a= torch.sigmoid(a)
print(a)

a= torch.tanh(a)
print(a)

d= torch.linspace(-1,1,10)
print(d)
print(torch.relu(d))

print(F.relu(d))
