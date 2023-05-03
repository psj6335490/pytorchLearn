import torch

a = torch.randn(2, 3)

print(a.type())  # 打印torch.FloatTensor
print(type(a))  # 打印<class 'torch.Tensor'>
print(isinstance(a, torch.FloatTensor))  # 打印True

print(isinstance(a, torch.cuda.FloatTensor))  # 打印False
# 将a放到GPU中
a = a.to(torch.device('cuda'))
# 或这样也可以
a = a.cuda()
print(isinstance(a, torch.cuda.FloatTensor))  # 打印True