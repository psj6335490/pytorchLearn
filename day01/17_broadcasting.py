import torch
 
# 假设得到一个feature map，维度为4,64,20,20(B,C,H,W)
fm = torch.zeros(4, 64, 20, 20)
print(fm.type())
 
# 要为每一个channel加上一个bias（每个channel对应一个卷积核的结果）
bias = torch.arange(64)
# 将LongTensor转换为FloatTensor
bias = bias.type(torch.FloatTensor)
print(bias.size())
# 我们要给每个channel对应的4张20*20的feature map的所有元素加上bias
# 首先我们要从最小（最小范围）的维度开始扩展
bias = bias.unsqueeze(-1).unsqueeze(-1)
print(bias.size())
# 在fm的channel后面有H和W两个维度，所以我们在bias后面添加两个维度
# 然后使用broadcasting
res = fm+bias
print(res.size())
# print(res)