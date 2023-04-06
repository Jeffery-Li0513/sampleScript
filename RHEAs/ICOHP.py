'''
处理ICOHP数据
'''

from pymatgen.electronic_structure.cohp import *
from pymatgen.io.lobster.outputs import Icohplist,Cohpcar
import itertools
import re


elemnts = ['Nb', 'Mo', 'Ta', 'W']
bonds = [sorted(list(x)) for x in list(itertools.combinations_with_replacement(elemnts, 2))]
bonds_number = {tuple(key): [] for key in bonds}                 # 储存每种键的ICOHP值，fromkeys()方法共享内存，不能用

# 首先从ICOHPLIST文件中找出所有的同类化学键
complete_cohp = CompleteCohp.from_file('LOBSTER',filename='COHPCAR.lobster',structure_file='POSCAR')
icohp = Icohplist('ICOHPLIST.lobster').icohpcollection
list_atom1 = icohp._list_atom1
list_atom2 = icohp._list_atom2
list_atom = [sorted(list(x)) for x in zip(list_atom1, list_atom2)]          # 保存ICOHP中原子对的的列表
print('This system contains {} bondpairs.'.format(len(list_atom)))
for i in range(len(list_atom)):
    #去除每个原子对中元素后面的数字
    bondpair = [re.sub('[\d]','',x) for x in list_atom[i]]
    if bondpair in bonds:               # 判断该键是否在bonds列表中，将该键的ICOHP值保存到bonds_number字典中
        fermi_icohp = get_integrated_cohp_in_energy_range(complete_cohp,'{}'.format(i+1)).__float__()   # 获取积分到费米能级的icohp值
        bonds_number[tuple(bondpair)].append(round(fermi_icohp,5))          # round控制浮点数输出精度

all_bonds = []
for key in bonds_number.keys():
    print('The total number of {} is {}'.format(key, len(bonds_number[key])))
    if len(bonds_number[key]) != 0:
        average_iCOHP = round((sum(bonds_number[key]) / len(bonds_number[key])), 5)
    else:
        average_iCOHP = None
    print('The average iCOHP of {} is {}'.format(key, average_iCOHP))

    all_bonds += bonds_number[key]

total_iCOHP = round((sum(all_bonds) / len(all_bonds)), 5)
print('The total iCOHP is {}'.format(total_iCOHP))