'''
计算优化前和优化后结构之间的均方根位移，注意考虑周期性边界条件

计算方法：
每一对原子距离的平方和，求和

usage：python meanSquareError.py POSCAR CONTCAR
'''

import numpy as np
import sys
from ase.io.vasp import read_vasp

# # 读取结构文件，注意这里默认每个原子在结构文件中的顺序是一样的
# def read_POSCAR(file):
#     '''
#     返回原子坐标列表
#     :param file:
#     :return:
#     '''
#     atoms = []
#     with open(file, 'r') as f1:
#         POSCAR = f1.readlines()
#         # 读取坐标信息
#         if POSCAR[7].strip() == "Selective dynamics":
#             coordiate = POSCAR[9:]
#         else:
#             coordiate = POSCAR[8:]
#         for i in coordiate:
#             one_coor = i.strip().split()[:3]
#             if one_coor:
#                 one_coor = [float(x) for x in one_coor]
#                 atoms.append(one_coor)
#     return atoms

# 计算两点之间的距离，考虑周期性边界条件
def distance(atoms1, atoms2, cell):
    dx = (atoms1[0] - atoms2[0])
    if (abs(dx) > cell[0]*0.5):
        dx = cell[0] - abs(dx)
    dy = (atoms1[1] - atoms2[1])
    if (abs(dy) > cell[1]*0.5):
        dy = cell[1] - abs(dy)
    dz = (atoms1[2] - atoms2[2])
    if (abs(dz) > cell[2]*0.5):
        dz = cell[2] - abs(dz)
    dis = np.sqrt(dx**2 + dy**2 + dz**2)
    # print("{}\t{}\t{}".format(dx, dy, dz))
    return dis

# 计算均方根位移
def MSE(atoms_1, atoms_2, cell):
    mse = 0
    if len(atoms_1) == len(atoms_2):
        for i in range(len(atoms_1)):
            # vector = np.array(atoms_1[i], dtype=float) - np.array(atoms_2[i], dtype=float)
            # length = np.linalg.norm(vector)
            length = distance(atoms_1[i], atoms_2[i], cell)
            print(length)
            mse += length
        return mse / len(atoms_1)
    else:
        print("两个结构原子数不一致")


if __name__ == '__main__':
    poscar_1 = read_vasp(file=sys.argv[1])
    atoms_1 = poscar_1.positions                        # 都是笛卡尔坐标
    poscar_2 = read_vasp(file=sys.argv[2])
    atoms_2 = poscar_2.positions
    mse = MSE(atoms_1, atoms_2, poscar_2.cell.lengths())
    print(mse)