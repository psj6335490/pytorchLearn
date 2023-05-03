import torch
import numpy as np

# linspace将[0,10]等分，steps表示数量（非步长）
aa = torch.linspace(0,10,steps=4)
print(aa) # 打印tensor([0.0000, 3.3333, 6.6667, 10.0000])
bb = torch.linspace(0,10,steps=11)
print(bb)
# 将[0,1]分成10个数n，算base的n次方
cc = torch.logspace(0,1,steps=10,base=2)
print(cc) # 输出tensor([1.0000, 1.0801, ... ,2.0000])
dd = torch.logspace(0,-1,steps=10)
print(dd)
 
# [0,10)之间等差数列，step为步长
ee = torch.arange(0,10,step=2)
print(ee) # 输出tensor([0,2,4,6,8])