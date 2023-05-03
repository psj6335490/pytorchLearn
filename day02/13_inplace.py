import  torch
import torch.nn as nn

x=torch.randn(2,2)

layer=nn.ReLU(inplace=False)
out=layer(x)

pass