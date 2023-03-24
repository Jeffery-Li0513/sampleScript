# -*- coding: utf-8 -*-
# @Time    : 2023/1/4 21:11
# @Author  : zefengLi
# @email   : by2201136@buaa.edu.cn
# @Comment : 支持向量机分类

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from test import load_data

(train_data, train_label), (test_data, test_label) = load_data("恒星/train.csv")

# 训练SVM分类器
# classifier = svm.SVC(C=0.5, kernel='linear', gamma='auto', decision_function_shape='ovr')
# classifier.fit(train_data, train_label.ravel())
# print("训练集：", classifier.score(train_data, train_label))
# print("测试集：", classifier.score(test_data, test_label))

print(np.arange(1, 11, 1))
