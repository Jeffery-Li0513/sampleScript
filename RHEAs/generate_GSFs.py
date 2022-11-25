'''
利用atomsk生成广义层错

0、对slab模型进行优化
1、在当前目录下准备好POSCAR文件，首先得到产生层错对应的参数，提供给atomsk
2、管道调用atomsk
3、提交计算任务

该脚本在pbs脚本中运行
'''

import subprocess
from subprocess import PIPE
import sys
import os

pbs_nodefile = sys.argv[1]
np = sys.argv[2]


layer = 12                      # slab模型的层数
z_length = 32.87314                 # z轴长度
move_array = [-0.80869, 0.57183, 0.0]                 # 移动的方向矢量，[0.0, 1.14365625, 0.0]
first_layer = 2.34903              # 第一层的z坐标
secend_layer = 4.57505             # 第二层的z坐标
sample = 6                        # 采样的构型数


# 判断需要的参数是否设置
if (layer == None) or (z_length == None) or (move_array == []) or (first_layer == None) or (secend_layer == None):
    print("参数错误")
    sys.exit()

# 需要移动哪几层，前两层和最后两层不要
interval = (secend_layer - first_layer)                # 每层之间的间隔
interval_1_2 = (secend_layer - first_layer) / 2        # 第一层与第二层中间位置
move_layer = []                                        # 储存需要移动的层间坐标
for i in range(2,layer-2):
    move_layer.append(interval_1_2 + interval * i)


# k为第k个采样的构型
for k in range(sample):
    for i in range(len(move_layer)):
        command_1 = "mkdir {} && cp POSCAR ./{}".format(i, i)
        command_2 = "(echo n; echo POSCAR-move) | atomsk POSCAR -select above {}*box Z -shift {} {} {} vasp;\
         cp POSCAR-move POSCAR".format(move_layer[i] / z_length, move_array[0], move_array[1], move_array[2])
        command_3 = "cp INCAR-static KPOINTS ./{}/{}".format(k, i)
        command_4 = "mv INCAR-static INCAR; vaspkit -task 103; mpirun -machinefile {} -np {} vasp_std > output"\
            .format(pbs_nodefile, np)
        p1 = subprocess.Popen(command_1, shell=True, cwd=os.getcwd() + '/{}'.format(k))
        p1.wait()
        p2 = subprocess.Popen(command_2, shell=True, stdout=PIPE, cwd=os.getcwd() + '/{}'.format(k) + '/{}'.format(i))
        p2.wait()
        p3 = subprocess.Popen(command_3, shell=True, cwd=os.getcwd())
        p3.wait()
        p4 = subprocess.Popen(command_4, shell=True, stdout=PIPE, cwd=os.getcwd() + '/{}'.format(k) + '/{}'.format(i))
        p4.wait()
