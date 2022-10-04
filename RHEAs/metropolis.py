'''

'''

import numpy as np
import math
import sys


def metropolis(E1, E2, T):
    '''
    判断是否能够进行交换，每交换一对原子都进行一次判定。如果不进行交换，就不用导出新的结构文件，直接将原来的结构文件复制过去即可。
    :return: True or False
    '''
    k = 8.617333262145E-5                       # 玻尔兹曼常数，eV/K 单位
    random_number = np.random.rand()             # 生成0-1之间的随机数
    delta_E = E2 - E1
    if (E1 > E2):
        return True
    possibility = math.exp(- delta_E / (k*T))
    if (possibility > random_number):
        return True
    return False


if __name__ == '__main__':
    E1 = float(sys.argv[1])
    E2 = float(sys.argv[2])
    T = float(sys.argv[3])
    print(metropolis(E1=E1, E2=E2, T=T))
