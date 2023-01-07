# -*- coding: utf-8 -*-
# @Time    : 2023/1/4 21:54
# @Author  : zefengLi
# @email   : by2201136@buaa.edu.cn
# @Comment : 神经网络实现分类任务
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
from torch.autograd import Variable
from test import load_data

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        return out

class Dataset(data.Dataset):
    def __init__(self, data_dir, filename):
        # self.file_name = os.listdir(data_dir)
        self.file_name = filename
        self.data_path = []
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(data_dir, self.file_name))

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        data = pd.read_csv(self.data_path[index])      # 读取每一个数据
        data.astype(float)
        data = torch.tensor(data.values)                            # 转成张量
        return data

# 读取数据
data_dir = "D:\Document\git\ML\恒星"
filename = "train.csv"
train_dataset = Dataset(data_dir=data_dir, filename=filename)
train_iter = data.DataLoader(train_dataset)

print(len(train_iter))


# net = Net(16, 20, 3)
# print(net)
#
# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# loss_func = nn.CrossEntropyLoss()
#
# (train_data, train_label), (test_data, test_label) = load_data("恒星/train.csv")
# train_data_tensor = torch.from_numpy(train_data)
# train_label_tensor = torch.from_numpy(train_label)
# train_data_tensor = train_data_tensor.to(torch.float32)
# train_label_tensor = train_label_tensor.to(torch.long)
#
# for t in range(100):
#     prediction = net(train_data_tensor)
#     loss = loss_func(prediction, train_label_tensor)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if t % 2 == 0:
#         plt.cla()
#         # 过了一道 softmax 的激励函数后的最大概率才是预测值
#         # print(F.softmax(prediction))
#         prediction = torch.max(F.softmax(prediction), 1)[1]
#         pred_y = prediction.data.numpy().squeeze()
#         target_y = train_label_tensor.data.numpy()
#         plt.scatter(train_data_tensor.data.numpy()[:, 0], train_data_tensor.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
#         plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()