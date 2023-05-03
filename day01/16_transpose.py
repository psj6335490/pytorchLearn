import torch
 
a = torch.rand(3, 4)
 
# a的转置
a_t = a.t()
print(a_t.size())
 
### 使用transpose交换维度
# 假设b代表4张mnist图片，维度分别代表B,C,H,W
b = torch.rand(4, 1, 28, 28)
# 将b的C和W维度交换，得到的维度为B,W,H,C
b_trans = b.transpose(1, 3)
print(b_trans.size())  # 输出torch.Size([4,28,28,1])
 
# 在交换维度后，需要随时用contiguous()来将数据重新归为连续状态
c = torch.rand(4, 3, 32, 32)
# 交换维度，然后使之连续，然后调整维度，然后再交换回来，看c和d是否一致
d = c.transpose(1, 3).contiguous().view(4, 32, 32, 3).transpose(1, 3)
# 如果输出为1，则表示c和d数据相同
print(torch.all(torch.eq(c, d)))
 
### 使用permute()直接调整所有维度的顺序
# 将维度变为H,W,C,B
e = c.permute(2,3,1,0)
print(e.size())