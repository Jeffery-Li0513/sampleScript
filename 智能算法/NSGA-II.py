# 实现NSGA-II遗传算法，进行多目标优化multi-objects optimizetion（MOP）

import math
import numpy as np
import random
import matplotlib.pyplot as plt

# 定义需要被优化的目标函数
def function1(x):
    value = -x**2
    return value
def function2(x):
    value = -(x-2)**2
    return value

# 找到列表值对应的指数，但是如果有重复元素，每次只能找出第一个
def index_of(a, list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

# 给个体排序，这个函数的目的是什么
def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
        # math.inf是一个常量，表示正无穷大，进行一次循环后将最小值设为无穷大，下一次就不会找到它。
    return sorted_list

# 实现快速非支配排序，最后返回的是一组非支配解，即Pareto前沿
def fast_non_dominated_sort(values1, values2):
    # values1和values2中储存的是种群对应于两个目标函数的值
    S = [[] for i in range(0, len(values1))]                # S用来记录种群中被个体p支配的个体的集合，因为有20个体，所以会有20个[]
    front = [[]]                                            # 储存非支配解，以个体对应指数的形式储存
    n = [0 for i in range(0,len(values1))]                  # n用来记录种群中支配个体p的个体数
    rank = [0 for i in range(0,len(values1))]               # 储存每个个体的级别
    # 对种群中每一个个体，都进行n和s集合的计算
    for p in range(0,len(values1)):
        S[p] = []               # 最初的时候，对于每一个个体，S[p]都是空的
        n[p] = 0                # 最初的时候，对每一个个体，n[p]都是0
        for q in range(0,len(values1)):
            # 对每一个个体p都遍历种群，找出其支配的个体，存到S中，找到支配p的个体，将其数量存在n中
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                # 这个判断关系是说 p支配q，因此可以将q存在S[p]中，前提是满足单一性。
                if q not in S[p]:
                    S[p].append(q)
            else:
                # 否则的话就是p被支配，支配p的个体数加1
                n[p] = n[p] + 1
        if n[p] == 0:
            # 如果n[p]=0，即没有支配p的个体，说明p是一个非支配解，将它的rank的级别设为最低，这样后面的虚拟适应度才会高
            rank[p] = 0
            if p not in front[0]:
                # 同时将此解加入到Pareto前沿中，此时front[0]中都是最开始产生的非支配解，我们可以将其称之为F1
                front[0].append(p)

    i = 0
    # 记住这个循环条件，是用来判断整个种群有没有被全部分级的
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                # 对于当前集合F1中的每一个个体p及其所支配的的个体集合，遍历S[p]中的每一个个体q，执行n[q]-1，如果此时n[q]=0，就将个体q保存在集合Q中，同时q的等级+1
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        # 记F1中得到的个体，即front[0]中的个体为第一个非支配层的个体，并以Q为当前集合，重复上述操作，一直到整个种群被分级
        front.append(Q)

    del front[len(front)-1]         # 删掉最后一个元素？为什么，最后一个里面的不算非支配解
    return front

# 计算拥挤因子
def crowding_distance(values1, values2, front):
    # 计算属于同一个非支配层的两个个体之间的拥挤度，这里传入的front参数应该某一支配层的，不是所有的
    # 先取出当前非支配层的所有个体，一共有len(front)
    distance = [0 for i in range(0,len(front))]
    # 对于每一个目标函数
    # （1）基于该目标函数对种群进行排序
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    # （2）令边界的两个个体拥挤度无穷
    distance[0] = np.inf
    distance[len(front)-1] = np.inf
    # （3）distance[k]从第二个个体开始，计算distance[k] = distance[k] + [(fm(i+1)-fm(i-1))] / (max-min)
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + ((values1(sorted1[k+1])-values1(sorted1[k-1])) / (max(values1)-min(values1)))**2
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + ((values2(sorted2[k+1])-values2(sorted2[k-1])) / (max(values2)-min(values2)))**2
    return distance**0.5

# 交叉算子
def crossover(a,b):
    r = random.random()
    if r > 0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

# 变异算子，即突变
def mutation(solution, max_x, min_x):
    mutation_prob = random.random()     # 变异概率
    if mutation_prob < 0.5:             # 突变率
        solution = solution + (max_x-min_x)*random.random()
    else:
        solution = solution
    return solution

