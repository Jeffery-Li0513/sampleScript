'''
求解a+b+c+d=100的所有解, a,b,c,d均为整数, 且a,b,c,d均大于等于5, 小于等于35。得到的解保存到all_solutions.txt文件中。
'''

import numpy as np


def solution():
    '''
    求解a+b+c+d=100的所有解, a,b,c,d均为整数, 且a,b,c,d均大于等于5, 小于等于35。得到的解保存到all_solutions.txt文件中。
    '''
    all_solutions = []
    a = np.arange(1, 98)
    b = np.arange(1, 98)
    c = np.arange(1, 98)
    d = np.arange(1, 98)
    for i in a:
        for j in b:
            for k in c:
                for l in d:
                    if i + j + k + l == 100:
                        # print(i, j, k, l)
                        all_solutions.append([i, j, k, l])
    print(len(all_solutions))
    np.savetxt('all_solutions_1_97.txt', all_solutions, fmt='%d')

def random_select(num):
    '''
    从all_solutions.txt文件中随机抽取num个解, 并将这10个解保存到random_solutions.txt文件中。
    '''
    all_solutions = np.loadtxt('all_solutions_1_97.txt', dtype=np.int32)
    random_solutions = np.random.choice(all_solutions.shape[0], num, replace=False)
    np.savetxt('random_solutions_500.txt', all_solutions[random_solutions], fmt='%d')


def remove_and_select(num, old_selected):
    '''
    从all_solutions.txt文件中删除重复的解。
    '''
    all_solutions = np.loadtxt('all_solutions_1_97.txt', dtype=np.int32)
    have_selected = np.loadtxt(old_selected, dtype=np.int32)
    all_solutions = np.delete(all_solutions, have_selected, axis=0)
    random_solutions = np.random.choice(all_solutions.shape[0], num, replace=False)
    np.savetxt('random_solutions_{}.txt'.format(num), all_solutions[random_solutions], fmt='%d')

if __name__ == '__main__':
    # solution()
    # random_select(500)
    remove_and_select(1000, old_selected='random_solutions_500.txt')