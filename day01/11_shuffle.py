import torch
 
a = torch.rand(10, 3)
b = torch.rand(10, 2)
print('a:', a)
print('b:', b)
 
# 产生一个随机顺序的index向量，根据需要shuffle的实际数据的维度
idx = torch.randperm(10)
print('idx:', idx)  # 这里输出的是[0,10)的一维向量，顺序是乱的
 
# 用同一个随机种子做shuffle，如果需要shuffle顺序不同，则需要产生不同的idx
a = a[idx]  # 相当于做了shuffle
b = b[idx]  # 相当于做了shuffle
print('a after shuffle:', a)
print('b after shuffle:', b)