import torch
 
### 高级操作where，可以实现高度并行的赋值
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
 
# 我们使用一个condition矩阵来决定取a和b中的哪些值来组成c
cond = torch.ByteTensor([[0, 1], [1, 0]])
# 通过cond来选择每一个元素从a还是b中获得，1表示a，0表示b
c = torch.where(cond, a, b)
print(c)
 
# 还可以这样用
cond2 = torch.rand(2, 2)
c2 = torch.where(cond2 > 0.5, a, b)
print(c2)
 
### 高级操作gather，实现查表
# 假设33是dog，44是cat，55是fish
table = torch.tensor([33, 44, 55])
# 假设我有一个向量，所有元素都是0,1,2。对应table中dim=0的3个index
find_list = torch.tensor([2, 1, 2, 0, 0, 1, 2])
found_in_table = torch.gather(table, dim=0, index=find_list)
print(found_in_table)  # 输出tensor([55,44,55,33,33,44,55])
print("################################################################33")
# 也可以是多维的
table2 = torch.rand(4, 10)
print(table2)
find_list2 = torch.randint(0, 10, [4, 5])
print(find_list2)
# 在每一行中获取5个index对应的值
found_in_table2 = torch.gather(table2, dim=1, index=find_list2)
print(found_in_table2)  # 输出一个4*5的矩阵，其中的值都来自于table2