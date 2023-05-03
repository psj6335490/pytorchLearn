import torch
 
a = torch.rand(3, 4)
b = torch.rand(4)
 
### 基本运算
# a+b broadcasting
ab_sum1 = a + b
ab_sum2 = torch.add(a, b)
print(torch.all(ab_sum1.eq(ab_sum2)))
# a-b broadcasting
ab_sub1 = a - b
ab_sub2 = torch.sub(a, b)
print(torch.all(ab_sub1.eq(ab_sub2)))
# a*b broadcasting
ab_mul1 = a * b
ab_mul2 = torch.mul(a, b)
print(torch.all(ab_mul1.eq(ab_mul2)))
# a/b broadcasting
ab_div1 = a / b  # 整除用//
ab_div2 = torch.div(a, b)
print(torch.all(ab_div1.eq(ab_div2)))
 
### 矩阵乘法
c = torch.rand(2, 3)
d = torch.rand(3, 4)
# 矩阵乘法的三种方式，推荐第二种，即matmul()和第三种@
cd_mm1 = torch.mm(c, d)
cd_mm2 = torch.matmul(c, d)
cd_mm3 = c @ d
print(torch.all(cd_mm1.eq(cd_mm2)))
print(torch.all(cd_mm2.eq(cd_mm3)))

print("超过二维的矩阵乘法")
### 超过二维的矩阵乘法
e = torch.rand(4, 3, 28, 64)
f = torch.rand(4, 3, 64, 32)
# 只针对最后两维做乘法，前面的两维至少要满足能够broadcasting
ef_mm = e @ f
print(ef_mm.size())  # 输出torch.Size([4,3,28,32])
 
g = torch.rand(4, 1, 64, 32)
# 这里的第二个维度使用了broadcasting
eg_mm = e @ g
print(eg_mm.size())  # 输出torch.Size([4,3,28,32])
 
### 错误示范
# h = torch.rand(4, 64, 32)
# # 由于无法执行broadcast，报错
# eh_mm = e @ h
# print(eh_mm.size())
 
 
aa = torch.full([3, 3], 10)
### N次方
# 使用以下两种方式计算N次方
print(aa.pow(2))
print(aa ** 3)
 
### 平方根
print(aa.sqrt())
# 平方根的倒数
print(aa.rsqrt())
# 开三次方
print(aa ** (1 / 3))


### exp
bb = torch.exp(aa)
print(bb)
 
### log
a_log10 = torch.log10(aa)
a_log2 = torch.log2(aa)
b_log = torch.log(bb)  # 以e为底
print(a_log10)
print(a_log2)
print(b_log)
 
### 向上向下取整
aaa = torch.randn(2, 3)
a_floor = aaa.floor()  # 向下取整
a_ceil = aaa.ceil()  # 向上取整
print(a_floor)
print(a_ceil)



print("截取整数和小数")
print(aaa)
### 截取整数和小数
a_trunc = aaa.trunc()  # 截取整数部分
a_frac = aaa.frac()  # 截取小数部分
print(a_trunc)
print(a_frac)
 
### 四舍五入
a_round = aaa.round()
print(a_round)
 
### 最大值最小值，中值，平均
grad = torch.randn(2, 3) * 15
print(grad)
print(grad.max())  # 最大值
print(grad.min())  # 最小值
print(grad.mean())  # 平均值
print(grad.median())  # 中间值
print(grad.prod()) # 所有元素累乘
print(grad.sum()) #所有元素求和
# 将小于5的数全部置为5，大于5的数不变
print(grad.clamp(5))
# 将数值全部限定在0-10范围，大于10的取10，小于0的取0.
print(grad.clamp(0, 10))