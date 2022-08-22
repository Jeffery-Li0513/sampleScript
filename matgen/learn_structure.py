from pymatgen.core import Structure, sites
import re
from pymatgen.io.lammps.data import LammpsData

# # pymetgen测试
# structure = Structure.from_file('POSCAR-5')
# # print(structure)
# print(structure.sites)
# print(structure[41])
# print(type(structure[41]))
# print(structure[41]._species)               # 返回元素名称
# print(structure.composition.formula)        # 返回一个字符串，储存元素名称和对应的原子个数————注意不是按照POSCAR中的顺序排列的
# print(len(structure))                       # 结构中的原子数
# print(structure.composition.element_composition)        # 这里面的组成是按照POSCAR顺序排列的
# print(type(structure.composition.element_composition))
# print(type(structure.species[0]))
# print([structure.species[i].symbol for i in range(len(structure))])         # 输出元素名称
#
# # 测试structure中的交换原子位置
# structure.replace(41, 'Nb')
# print(structure[41])
#
# # elements = structure.composition.formula.split()
# p = re.compile('(\D+)(\d+)')
# # elements = [p.match(i).groups() for i in structure.composition.element_composition.split()]
# elements_symbol = [k.symbol for k in structure.composition.element_composition.keys()]
# elements_number = [int(k) for k in structure.composition.element_composition.values()]
# elements = list(zip(elements_symbol, elements_number))
# print(elements)
#
# # 每种元素在坐标列表中对应的序号索引
# elements_index = {}
# a = 0
# for i in elements:
#     elements_index[i[0]] = [a, a+int(i[1])]
#     a += int(i[1])
# print(elements_index)
#
# a = [i for i in range(10)]
# print(a[10:])


# 测试lammps结构文件的读取
structure = LammpsData.from_file('NbMoTaW.data', atom_style='atomic')           # 需要用到lammps.io.lammps下面的相关类
print(structure.atoms)                  # 所有原子的坐标和类型，以pandas的DataFrame形式储存
print(type(structure.atoms))
print(len(structure.atoms))
print(structure.atoms.iloc[0]['type'])
print(structure.masses)                     # 原子质量
print(structure.box)                        # 模拟盒子的尺寸
# print(structure.velocities)                 # 所有原子的速度信息
print(type(structure.velocities))
print(structure.force_field)                # 力场的信息
print(structure.topology)                   # 这是什么？包含一些键角键长等结构数据
print(structure.atom_style)                 # atom_style参数

# new_struture = structure.copy()
structure.atoms.iloc[0, 0] = 3
structure.write_file('new_structure.data')