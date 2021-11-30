# coding=utf8
'''
Author: Creling
Date: 2021-11-29 21:13:43
LastEditors: Creling
LastEditTime: 2021-11-29 23:19:16
Description: file content
'''
'''
torchvision包是服务于Pytorch深度学习框架的，用来生成图片、视频数据集和一些流行的模型类和预训练模型
torchvision 由以下四个模块组成
torchvision.datasets
torchvision.models
torchvision.transforms
torchvision.utils
'''
# 变量

import torch
from torch.autograd import Variable
x = torch.ones((2, 2), requires_grad=True)
y = torch.ones((2,2), requires_grad=True) 
print(x)
print(y)
z = 2*x + 3*y
out = z.mean()
out.backward()
print(x.grad)
print(y.grad)
# # 如果requires_grad=True，在进行反向传播的时候会记录该tensor梯度信息。
# x = Variable(torch.ones(2, 2), requires_grad=True)
# print(x)

# y = x + 2
# print(y)
# z = y * y * 3
# out = z.mean()
# print(z)
# print(out)

# out.backward()
# # out是标量，因此不需要为backward()函数指定参数,相当于out.backward(torch.tensor(1))
# print(x.grad)  # 打印梯度d(out)/dx

# x = torch.randn(3)

# x = Variable(x, requires_grad=True)
# y = x * 2
# while y.data.norm() < 500:
#     y = y * 2
# print(y)

# gradients = torch.FloatTensor([0.1, 1.0, 0.001])
# y.backward(gradients)
# print(x.grad)
