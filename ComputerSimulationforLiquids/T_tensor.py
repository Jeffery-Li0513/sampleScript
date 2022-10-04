'''
通过stdin输入时，需要在输入字典后先输enter键，然后ctrl+d

'''

import numpy as np
from itertools import product
import json
import sys
from maths_module import random_vector

def t2_tensor(r, r3):
    '''
    返回一个3x3的2阶T张量
    :param r:距离矢量
    :param r3:r模长的3次方
    :return:二阶T张量，一阶T张量就是r的倒数
    '''
    t2 = 3.0 * np.outer(r, r)                   # 两个矢量的外积
    t2 = t2 - np.identity(3)                    # np.identity(3)生成一个3维单位矩阵
    t2 = t2 / r3                                # 二阶T张量的计算公式，见书公式1.17

    return t2

def t3_tensor(r, r4):
    '''
    返回一个3x3x3的相互作用张量
    :param r:
    :param r4:
    :return:
    '''
    t3 = 15.0 * np.einsum('i,j,k->ijk', r, r, r)            # 三个矢量的外积——3x3x3矩阵
    for i in range(3):
        t3[i, i, i] = t3[i, i, i] - 9.0 * r[i]              # 修改三维索引都相同的元素
        for j in range(3):
            if j == i:
                continue
            t3[i, i, j] = t3[i, i, j] - 3.0 * r[j]
            t3[i, j, i] = t3[i, j, i] - 3.0 * r[j]
            t3[j, i, i] = t3[j, i, i] - 3.0 * r[j]
    t3 = t3 / r4

    return t3

def t4_tensor(r, r5):
    # 定义一个3x3的单位矩阵。作用类似克罗内克函数，指数不相同的元素为0
    u = np.identity(3)

    t4 = 105.0 * np.einsum('i,j,k,l->ijkl', r, r, r, r)
    for i,j,k,l in product(range(3), repeat=4):         # 类似嵌套的for循环
        t4[i,j,k,l] = t4[i,j,k,l] - 15.0 * (
            r[i] * r[j] * u[k,l] + r[i] * r[k] * u[j,l]
            + r[i] * r[l] * u[j,k] + r[j] * r[k] * u[i,l]
            + r[j] * r[l] * u[i,k] + r[k] * r[l] * u[i,j]
        ) + 3.0 * (u[i,j] * u[k,l] + u[i,k] * u[j,l] + u[i,l] * u[j,k])
    t4 = t4 / r5

    return t4

def t5_tensor ( r, r6 ):
    """Returns fifth-rank 3x3x3x3x3 interaction tensor
    Supplied arguments should be the unit vector from 2 to 1 and
    the sixth power of the modulus of that vector.
    """

    # 定义一个3x3的单位矩阵。作用类似克罗内克函数，指数不相同的元素为0
    u = np.identity(3)

    t5 = 945.0 * np.einsum('i,j,k,l,m->ijklm',r,r,r,r,r)                # 五个矢量的外积

    for i,j,k,l,m in product ( range(3), repeat=5 ):
        t5[i,j,k,l,m] = t5[i,j,k,l,m] - 105.0 * (
            r[i] * r[j] * r[k] * u[l,m] + r[i] * r[j] * r[l] * u[k,m]
            + r[i] * r[j] * r[m] * u[k,l] + r[i] * r[k] * r[l] * u[j,m]
            + r[i] * r[k] * r[m] * u[j,l] + r[i] * r[l] * r[m] * u[j,k]
            + r[j] * r[k] * r[l] * u[i,m] + r[j] * r[k] * r[m] * u[i,l]
            + r[j] * r[l] * r[m] * u[i,k] + r[k] * r[l] * r[m] * u[i,j]
            ) + 15.0 * (
            r[i] * ( u[j,k] * u[l,m] + u[j,l] * u[k,m] + u[j,m] * u[k,l] )
            + r[j] * ( u[i,k] * u[l,m] + u[i,l] * u[k,m] + u[i,m] * u[k,l] )
            + r[k] * ( u[i,j] * u[l,m] + u[i,l] * u[j,m] + u[i,m] * u[j,l] )
            + r[l] * ( u[i,j] * u[k,m] + u[i,k] * u[j,m] + u[i,m] * u[j,k] )
            + r[m] * ( u[i,j] * u[k,l] + u[i,k] * u[j,l] + u[i,l] * u[j,k] )
            )

    t5 = t5 / r6 # Scale by sixth power of distance

    return t5

def skew(a):
    '''
    实现两个向量的叉乘，传入的a为一个Levi-Civita张量，返回叉乘结果
    :param a: Levi-Civita tensor，两个index分别代表A、B向量的第i个元素
    :return:
    '''
    b = np.empty(3, dtype=np.float_)
    b[0] = a[1,2] - a[2,1]
    b[1] = a[2,0] - a[0,2]
    b[2] = a[0,1] - a[1,0]
    return b

print('t_tensor')
print('Calculation of electrostatic interactions between linear molecules')
print('using T-tensors and Euler angles')

# 以json格式从命令行读取参数，如果解析失败就退出程序
try:
    nml = json.load(sys.stdin)
except json.JSONDecodeError:
    print('Exiting on Invalid JSON format')
    sys.exit()

# 设置默认值，并检查命令行输入的键和检查值的类型
defaults = {"d_min":0.5, "d_max":1.5, "mu1_mag":1.0, "mu2_mag":1.0, "quad1_mag":1.0, "quad2_mag":1.0}
for key,val in nml.items():
    if key in defaults:
        assert type(val) == type(defaults[key]), key+"has the wrong type"           # 在表达式条件为false时输出后面的语句
    else:
        print('Warning', key, 'not in ', list(defaults.keys()))

# 将参数设为输入值还是默认值
d_min = nml["d_min"]        if "d_min" in nml else defaults["d_min"]                            # 最小间距
d_max = nml["d_max"]        if "d_max" in nml else defaults["d_max"]                            # 最大间距
mu1_mag = nml["mu1_mag"]        if "mu1_mag" in nml else defaults["mu1_mag"]                    # 分子1的偶极矩
mu2_mag = nml["mu2_mag"]        if "mu2_mag" in nml else defaults["mu2_mag"]                    # 分子2的偶极矩
quad1_mag = nml["quad1_mag"]        if "quad1_mag" in nml else defaults["quad1_mag"]            # 分子1的四极矩
quad2_mag = nml["quad2_mag"]        if "quad2_mag" in nml else defaults["quad2_mag"]            # 分子2的四极矩

# 输出参数
print ( "{:40}{:15.6f}".format('Min separation d_min',            d_min)      )             # :40应该是限制40个字符
print ( "{:40}{:15.6f}".format('Max separation d_max',            d_max)      )
print ( "{:40}{:15.6f}".format('Dipole moment of molecule 1',     mu1_mag)    )
print ( "{:40}{:15.6f}".format('Dipole moment of molecule 2',     mu2_mag)    )
print ( "{:40}{:15.6f}".format('Quadrupole moment of molecule 1', quad1_mag)  )
print ( "{:40}{:15.6f}".format('Quadrupole moment of molecule 2', quad2_mag)  )

np.random.seed()

# 随机选择方向，生成两个模长为1的三维随机向量
e1 = random_vector()
e2 = random_vector()

# 将原子2放在原点，原子1以随即方向和想要的距离放置
r12_hat = random_vector()                           # 沿着随机方向的单位向量
r12_mag = np.random.rand()
r12_mag = d_min + (d_max - d_min) * r12_mag         # 两原子间距离
r12 = r12_mag * r12_hat                             # 沿着r12_hat方向，长度为r12_mag

c1 = np.dot(e1, r12_hat)                            # e1和r12夹角的cos值，因为这俩方向向量模长都是1
c2 = np.dot(e2, r12_hat)                            # e2和r12夹角的cos值
c12 = np.dot(e1, e2)                                # e1和e2夹角的cos值

print ( "{:40}{:10.6f}{:10.6f}{:10.6f}".format('Displacement r12', *r12)  )         # 列表或元组变量前面加星号，表示拆解为独立元素传入
print ( "{:40}{:10.6f}{:10.6f}{:10.6f}".format('Orientation e1',   *e1)   )
print ( "{:40}{:10.6f}{:10.6f}{:10.6f}".format('Orientation e2',   *e2)   )

# 固定坐标系中的偶极向量
mu1 = mu1_mag * e1
mu2 = mu2_mag * e2

# 固定坐标系中的四极张量
quad1 = 1.5 * np.outer(e1, e1)
quad1 = quad1 - 0.5*np.identity(3)
quad1 = quad1_mag * quad1
quad2 = 1.5 * np.outer(e2, e2)
quad2 = quad2 - 0.5*np.identity(3)
quad2 = quad2_mag * quad2

# 不同阶的T张量
tt2 = t2_tensor(r12_hat, r12_mag**3)
tt3 = t3_tensor(r12_hat, r12_mag**4)
tt4 = t4_tensor(r12_hat, r12_mag**5)
tt5 = t5_tensor(r12_hat, r12_mag**6)

# Heading
print()
print("{:>66}{:>40}{:>40}".format('.....Result from T tensor','.....Result from Euler angles','.........Difference') )

print('\nDipole-dipole')
e_fmt = "{:30}{:36.6f}{:40.6f}{:40.2e}"                         # 能量和力的输出格式
f_fmt = "{:30}{:12.6f}{:12.6f}{:12.6f}{:16.6f}{:12.6f}{:12.6f}{:16.2e}{:12.2e}{:12.2e}"

# 计算偶极-偶极之间能量
v12t = -np.einsum('i,ij,j', mu1, tt2, mu2)
v12e = (mu1_mag * mu2_mag / r12_mag **3) * (c12 - 3.0 * c1 * c2)
print(e_fmt.format('Energy', v12t, v12e, v12t-v12e))

# 计算偶极-偶极之间的力

# 计算偶极-偶极之间的力矩

# 计算偶极-四极之间的能量

# 计算偶极-四极之间的力

# 计算偶极-四极之间的力矩

# 计算四极-偶极之间的能量

# 计算四极-偶极之间的力

# 计算四极-偶极之间的力矩

# 计算四极-四极之间的能量

# 计算四极-四极之间的力

# 计算四极-四极之间的力矩














