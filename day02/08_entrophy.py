import torch
import torch.nn.functional as F
x=torch.randn(1,784)
w=torch.randn(2,784)
logits=x@w.t()  # x和w矩阵相乘加上b得到logits
pred=F.softmax(logits,dim=1)  # 经过softmax得到一个pred
pred_log=torch.log(pred)  # 进行log操作，得到log（pred）
F.cross_entropy(logits,torch.tensor([1]))  # 使用cross entropy 第一项必须是logits，因为cross entropy内有softmax
#  python中cross entropy（）=softmax（）+log+null_loss（）
#输出tensor(2.9087)
F.nll_loss(pred_log,torch.tensor([1]))
#输出tensor(2.9087)