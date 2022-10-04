'''
利用ASE计算一系列结构的体积，并写入文件中
默认把所有子目录中的结构都找到，需要哪些数据自己处理
'''

from ase.io.vasp import read_vasp
import os

# 去除目录列表中的文件和不相关目录，并从小到大排序
dir_list = sorted([x for x in os.listdir() if (not x.isalpha()) and (os.path.isdir(x))])

with open('volume.txt', 'w') as f:
    f.write('strain\t\tvolume(angstrom^3)\n')
    for i in dir_list:
        if os.path.exists(i+'/CONTCAR'):
            contcar = read_vasp(file=i+'/CONTCAR')
            cell = contcar.cell
            f.write('{}\t\t{}\n'.format(i, cell.volume))
