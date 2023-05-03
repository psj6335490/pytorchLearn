import torch
 
a = torch.arange(12)
idx = torch.randperm(12)
a = a[idx]
a = a.view(3, 4).type(torch.float32)
print(a)
 
# 不带参数的argmax和argmin会把矩阵压平来返回index
print(a.argmax())
print(a.argmin())
 
# 如果想要在某个维度上使用argmax和argmin
# 返回每一列上最大值的index组成的向量，维度等于行的维度
print(a.argmax(dim=0))
# 获取每一列的最大值组成的向量,以及对应index组成的向量
print(a.max(dim=0))
# 返回每一行上最小值的index组成的向量，维度等于列的维度
print(a.argmin(dim=1))
# 获取每一行的最小值组成的向量,以及对应index组成的向量
print(a.min(dim=1))
 
### keepdim
# 返回的不是一个向量，返回保持是矩阵[3,4]--->[3,1]，而不是[3]
print(a.max(dim=1, keepdim=True).values.size())  # torch.Size([3,1])
 
### 获取topk
# 获取最大top2,[3,4]--->[3,2]
print(a.topk(2, dim=1))
# 获取最小top3,[3,4]--->[3,3]
print(a.topk(3, dim=1, largest=False))
 
### 获取第n小
# 获取每行第3小的数及index
print(a.kthvalue(3, dim=1))
# 获取每列第2小的数及index
print(a.kthvalue(2, dim=0))