import scipy.special as ssp             # 里面有激活函数sigmoid
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os, gzip
import time

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        '''
        用于类的初始值的设定
        :param inputnodes: 输入层节点数
        :param hiddennodes: 隐藏层节点数
        :param outputnodes: 输出层节点数
        :param learningrate: 学习率
        '''
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置输入层与隐藏层直接的权重关系矩阵以及隐藏层与输出层之间的权重关系矩阵
        # 一开始随机生成权重矩阵的数值，利用正态分布，均值为0，方差为隐藏层节点数的-0.5次方
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))    #矩阵大小为隐藏层节点数×输入层节点数
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))    #矩阵大小为输出层节点数×隐藏层节点数

        # 设置学习率
        self.lr = learningrate

        # 将激活函数sigmoid定义为self.activation_function
        self.activation_function = lambda x: ssp.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        '''
        数据集训练
        :param inputs_list: 神经网路的输入
        :param targets_list: 神经网络的输出，训练过程中为标签
        :return:
        '''

        # 将导入的输入数据和标签转换成二维矩阵
        inputs = np.array(inputs_list, ndim=2).T
        targets = np.array(targets_list, ndim=2).T

        # 进行前向传播
        hidden_inputs = np.dot(self.wih, inputs)
        # 利用激活函数sigmoid计算隐藏层输出的数据
        hidden_outputs = self.activation_function(hidden_inputs)
        # 利用隐藏层输出的数据计算导入输出层的数据
        final_inputs = np.dot(self.who, hidden_outputs)
        # 利用激活函数sigmoid计算输出层的输出结果
        final_outputs = self.activation_function(final_inputs)
        # 前向传播结束

        # 进行反向传播
        # 计算前向传播得到的输出结果与正确值之间的误差
        output_errors = targets - final_outputs
        # 隐藏层的误差是由输出层的误差通过两个层之间的权重矩阵进行分配的，在隐藏层重新结合
        hidden_errors = np.dot(self.who.T, output_errors)       # 隐藏层与输出层之间的权重矩阵的转置与前向传播的误差矩阵的点乘

        # 对隐藏层与输出层之间的权重矩阵进行迭代
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        # 对输入层与隐藏层之间的权重矩阵进行更新迭代
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

        pass

    def query(self, inputs_list):
        '''
        测试函数，
        :param inputs_list: 输入测试集数据
        :return: 预测测试标签
        '''

        inputs = np.array(inputs_list, ndim=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signal into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

def polt_confusion_matrix(cm, savename, title='Confusion Matrix'):
    '''
    定义画混淆矩阵的函数
    :param cm: 计算得到的混淆矩阵
    :param savename: 每个类的标签
    :param title: 标题
    :return:
    '''
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵种中每格的概率值
    ind_array = np.array(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=15, va='center', ha='center')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset of tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def load_data(data_folder):
    '''
    载入数据
    :param data_folder:数据来源
    :return: 训练集和测试集的数据及标签
    '''





















