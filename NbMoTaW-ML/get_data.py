'''
从计算数据中提取出所需的数据，生成数据集，数据集包含的数据有：
组分：Nb Mo Ta W
弹性常数分量：C11, C12, C44
晶格常数：a, b, c
剪切模量、体模量、杨氏模量、泊松比、B/G、Zener比
'''


import numpy as np
import pandas as pd
import os
import re
from ase.io import read

def get_data():
    '''
    从计算数据中提取出所需的数据，生成数据集
    '''
    # 定义储存数据的dataframe
    data = pd.DataFrame(columns=['formula', 'Nb', 'Mo', 'Ta', 'W', 'C11', 'C12', 'C44', 'a', 'b', 'c', 'G', 'B', 'E', 'v', 'Pugh', 'Zener'])
    # 获取目录列表
    path = os.getcwd()
    dir_list = []
    with open(path + '/random_solutions_1_97.txt') as f:
        solutions = f.readlines()
        for j in range(400):
            fraction = solutions[j].split()
            solution = 'Nb' + fraction[0] + 'Mo' + fraction[1] + 'Ta' + fraction[2] + 'W' + fraction[3]
            if os.path.isdir(os.path.join(path, solution)):
                dir_list.append(solution)

    print(len(dir_list))
    for i in dir_list:
        print(i)
        formula = i
        elements= re.findall(r'[A-Za-z]{1,2}', i)        # 获取元素
        fraction = re.findall(r'[0-9]{1,2}', i)         # 获取元素的含量
        Nb = int(fraction[0])
        Mo = int(fraction[1])
        Ta = int(fraction[2])
        W = int(fraction[3])
        with open(path + '/' + i + '/output-retart', 'r') as f:
            lines = f.readlines()
            if lines[-1].split()[0] == 'Total' and lines[-1].split()[1] == 'wall':
                C11 = round(float(lines[-10].split()[2]), 3)
                C12 = round(float(lines[-9].split()[2]), 3)
                C44 = round(float(lines[-8].split()[2]), 3)
                B = round(float(lines[-7].split()[3]), 3)
                G = round(float(lines[-6].split()[3]), 3)
                E = round(float(lines[-5].split()[3]), 3)
                v = round(float(lines[-4].split()[3]), 3)
                Pugh = round(float(lines[-3].split()[3]), 3)
                Zener = round(float(lines[-2].split()[3]), 3)
            else:
                continue
        structure = read(path + '/' + i + '/final.dump', format='lammps-dump-text')
        cell = structure.get_cell().cellpar()           # 获取晶格常数
        a = round(cell[0], 3)
        b = round(cell[1], 3)
        c = round(cell[2], 3)

        data.loc[len(data)] = [formula, Nb, Mo, Ta, W, C11, C12, C44, a, b, c, G, B, E, v, Pugh, Zener]        # 将数据添加到dataframe中
    print(len(data))
    data.sample(frac=1).reset_index(drop=True)          # 打乱数据集
    data.to_csv('data.csv', index=False)



if __name__ == '__main__':
    get_data()

