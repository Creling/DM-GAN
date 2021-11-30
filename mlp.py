# coding=utf8
'''
Author: Creling
Date: 2021-11-29 23:15:19
LastEditors: Creling
LastEditTime: 2021-11-30 09:31:32
Description: file content
'''

import torch
from tqdm import tqdm

M, input_size, hidden_size, output_size = 32, 1000, 100, 10
learning_rate = 0.001

x = torch.rand((M, input_size))
y = torch.rand((M, output_size))

w1 = torch.rand((input_size, hidden_size), requires_grad=True)
w2 = torch.rand((hidden_size, hidden_size), requires_grad=True)
w3 = torch.rand((hidden_size, output_size), requires_grad=True)
b1 = torch.rand((1, hidden_size), requires_grad=True)
b2 = torch.rand((1, hidden_size), requires_grad=True)
b3 = torch.rand((1, output_size), requires_grad=True)


def model(x, w1, w2, b1, b2, w3, b3):
    h1 = x.mm(w1) + b1
    h1 = h1.clamp(min=0)
    h2 = h1.mm(w2) + b2
    h2 = h2.clamp(min=0)
    output = h2.mm(w3) + b3
    return output


def loss_fn(y_pred, y):
    return (y - y_pred).pow(2).sum()


def train(x, w1, w2, b1, b2, w3, b3):
    for i in tqdm(range(5000)):
        output = model(x, w1, w2, b1, b2, w3, b3)
        loss = loss_fn(output, y)

        if i % 100 == 0:
            print("")
            print(loss.item())

        loss.backward()

        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data
        w3.data -= learning_rate * w3.grad.data
        b1.data -= learning_rate * b1.grad.data
        b2.data -= learning_rate * b2.grad.data
        b3.data -= learning_rate * b3.grad.data

        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()
        b3.grad.zero_()

print(w1)

train(x, w1, w2, b1, b2, w3, b3)

print(w1)