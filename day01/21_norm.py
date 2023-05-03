import torch
 
### 范数norm
a = torch.ones(8)
b = torch.ones(2, 4)
c = torch.ones(2, 2, 2)
 
print(a.norm(1), b.norm(1), c.norm(1))  # 8,8,8
print(a.norm(2), b.norm(2), c.norm(2))  # 2.8284,2.8284,2.8284
 
# 指定在哪一维上做norm
# 在b的dim=1上做L1范数
print(b.norm(1, dim=1))  # [4,4]
print(b.norm(2, dim=1))  # [2,2]
 
print(c.norm(1, dim=0))  # [[2,2],[2,2]]
print(c.norm(2, dim=0))  # [[1.4142,1.4142],[1.4142,1.4142]]