s = [[] for i in range(0, 5)]

# print(s)

import numpy as np
import random

n = 5
k = 2

# print(np.random.random(n) < 0.5)
# print(np.random.random([2,2]))              # 生成二维随机数
xs = np.random.random([2,5])
ys = np.random.random([2,5])
swaps = (np.random.random(n) < 0.5).repeat(k).reshape(k, n)
xs_ = np.select([swaps, ~swaps], [xs, ys])
ys_ = np.select([~swaps, swaps], [xs, ys])
# print(xs)
# print(ys)
# print(xs_)
# print(ys_)


# ll = np.random.random([3,4,2])
# print(ll)
# print(np.shape(ll))         # 方法
# print(ll.shape)             # 属性
# print(ll.shape[-1])
# print(len(ll))
# print(~True)

# front = [i for i in range(5)]
# print(front)
# del front[len(front)-1]
# print(front)

# ll = [-i for i in range(10)]
# print(ll)
# print(ll[:])
# print(min(ll))

mutation_prob = random.random()
# print(mutation_prob)

seq = ["a", 'b', 'c']
print(''.join(seq))