# -*- coding: utf-8 -*-
# @Time    : 2023/1/4 20:13
# @Author  : zefengLi
# @email   : by2201136@buaa.edu.cn
# @Comment : 随机森林分类

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(data_folder):
    '''
    载入数据
    :param data_folder:数据来源
    :return: 训练集和测试集的数据及标签
    '''
    tmp = np.loadtxt(data_folder, dtype=np.str, delimiter=",")
    data = tmp[1:, 1:-1].astype(np.float)
    label = tmp[1:, -1].astype(np.int)

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.3, random_state=7)

    return (train_data, train_label), (test_data, test_label)

(train_data, train_label), (test_data, test_label) = load_data("恒星/train.csv")

# 训练集和验证集大小
print("train data shape: {}".format(train_data.shape))
print("test data shape: {}".format(test_data.shape))

# 建立模型
rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(train_data, train_label)

# 查看模型效果
score_r = rfc.score(test_data, test_label)
print("Random Forest:", score_r)