import torch
 
a = torch.randn(3, 4)
print(a)
# 大于，满足的位置为1，不满足的位置为0
print(a > 0)
print(torch.gt(a, 0))
# 大于等于，同上
print(a >= 0)
print(torch.ge(a, 0))
# 小于，同上
print(a < 0)
print(torch.lt(a, 0))
# 小于等于，同上
print(a <= 0)
print(torch.le(a, 0))
# 不等于，同上
print(a != 0)
# 等于，同上
print(a == 0)
print(torch.eq(a, a))
 
# 判断是否一样，和上面的不一样
print(torch.equal(a, a))  # 输出True（和前面不一样）