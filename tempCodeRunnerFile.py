
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
