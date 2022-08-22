'''
基于metropolis算法的Monte Carlo模拟
'''

from pymatgen.core import Structure, sites
from pymatgen.io.lammps.data import LammpsData
import re
import numpy as np
import random
import math
import shutil


class ExchangeAtoms(object):
    def __init__(self, structure_path, exchange_times=1, exchange_species=None):
        '''

        :param structure_path: 读取输入的结构文件，支持lammps和vasp格式
        :param exchange_times: 每次交换多少对原子，默认交换一次
        :param exchange_species: 需要交换的原子种类，默认为所有元素都可以进行交换
        '''
        self.structure = Structure.from_file(structure_path)
        self.exchange_times = exchange_times
        self.exchange_species = exchange_species

    def chose_two_atom(self):
        '''
        选择两个不同种类的原子进行交换，需要识别exchange_species元素列表，只有在其中的才能交换
        :return: 返回要交换的两个原子的ID和种类
        '''
        # 首先处理读取到的结构数据
        elements_symbol = [k.symbol for k in self.structure.composition.element_composition.keys()]
        elements_number = [int(k) for k in self.structure.composition.element_composition.values()]
        elements = list(zip(elements_symbol, elements_number))
        # p = re.compile('(\D+)(\d+)')
        # elements = [p.match(i).groups() for i in self.structure.composition.element_composition.split()]      # [('Ta', '32'), ('Nb', '32'), ('Mo', '32'), ('W', '32')]
        # 每种元素在坐标列表中对应的序号索引
        elements_index = {}
        a = 0
        for i in elements:
            elements_index[i[0]] = [a, a+int(i[1])]
            a += int(i[1])
        # 原子的ID索引和对应的名称映射，从该列表中挑选需要交换的原子
        atom_index = [i for i in range(len(self.structure))]
        atom_name = [self.structure.species[i].symbol for i in range(len(self.structure))]
        # 储存进行交换的原子对，每一对的形式为 [('ID', 'species'), ('ID', 'species')]
        exchange_atoms = []
        for i in range(self.exchange_times):
            first_atom_ID = random.choice(atom_index)
            first_atom_specie = atom_name[first_atom_ID]
            # 从除了该元素的其他元素中抽取
            new_atom_index = atom_index[:elements_index[first_atom_specie][0]] + atom_index[elements_index[first_atom_specie][1]:]
            seconed_atom_ID = random.choice(new_atom_index)
            seconed_atom_specie = atom_name[seconed_atom_ID]
            atoms_pair = [(first_atom_ID, first_atom_specie), (seconed_atom_ID, seconed_atom_specie)]
            exchange_atoms.append(atoms_pair)
        return exchange_atoms

    def exchange(self, exchange_atoms):
        '''
        对原始结构structure进行位置替换
        :param exchange_atoms: 需要交换的原子列表
        :return: 返回交换后的structure
        '''
        for atom_pair in exchange_atoms:
            self.structure.replace(atom_pair[0][0], atom_pair[1][1])
            self.structure.replace(atom_pair[1][0], atom_pair[0][1])
        return
    def generate_new_structure(self):
        '''
        输出生成的新结构。
        :return:
        '''
        new_structure = self.structure.to('cif', 'structure.cif')
    def metropolis(self, E1, E2, T):
        '''
        判断是否能够进行交换，每交换一对原子都进行一次判定。如果不进行交换，就不用导出新的结构文件，直接将原来的结构文件复制过去即可。
        :return: True or False
        '''
        k = 8.617333262145E-5                       # 玻尔兹曼常数，eV/K 单位
        random_number = random.random()             # 生成0-1之间的随机数
        delta_E = E1- E2
        if (E1 > E2):
            return True
        possibility = math.exp(- delta_E / (k*T))
        if possibility > random_number:
            return True
        return False

class Lammps_MC(object):
    def __init__(self, structure_path, atom_style, exchange_speices=None):
        self.structure = LammpsData.from_file(structure_path, atom_style=atom_style)
        self.exchange_speices = exchange_speices
    def process_lammps_data(self, structure):
        '''
        处理lammps读取到的数据
        :param structure:
        :return:
        '''
        pass
    def chose_two_atoms(self):
        atoms = self.structure.atoms                # 以pandas Dataframe形式储存的原子信息
        # 选择一个原子
        first_atom_ID = np.random.choice(list(range(len(atoms))))
        first_atom_specie = atoms.iloc[first_atom_ID]['type']
        # 选择第二个原子
        while True:
            ID = np.random.choice(list(range(len(atoms))))
            if atoms.iloc[ID]['type'] != first_atom_specie:
                seconed_atom_ID = ID
                seconed_atom_specie = atoms.iloc[seconed_atom_ID]['type']
                break
        atoms_pair = [(first_atom_ID, first_atom_specie), (seconed_atom_ID, seconed_atom_specie)]
        return atoms_pair
    def exchange(self, atoms_pair):
        '''
        交换atoms原子中两个原子的类型，原子速度不用管
        :param atoms_pair:
        :return:
        '''
        self.structure.atoms.iloc[atoms_pair[0][0], 0] = atoms_pair[1][1]
        self.structure.atoms.iloc[atoms_pair[1][0], 0] = atoms_pair[0][1]
    def generate_new_structure(self):
        self.structure.write_file('new_structure.data')

def metropolis(E1, E2, T):
    '''
    判断是否能够进行交换，每交换一对原子都进行一次判定。如果不进行交换，就不用导出新的结构文件，直接将原来的结构文件复制过去即可。
    :return: True or False
    '''
    k = 8.617333262145E-5                       # 玻尔兹曼常数，eV/K 单位
    random_number = np.random.rand()             # 生成0-1之间的随机数
    delta_E = E1 - E2
    if (E1 > E2):
        return True
    possibility = math.exp(- delta_E / (k*T))
    print(possibility)
    if possibility > random_number:
        return True
    return False

if __name__ == '__main__':
    k = 0
    T = 300
    for i in range(10):
        mc = Lammps_MC('NbMoTaW_{}.data'.format(k), atom_style='atomic')
        atoms_pair = mc.chose_two_atoms()
        mc.exchange(atoms_pair=atoms_pair)
        E1 = np.random.random()
        E2 = np.random.random()
        print(metropolis(E1=E1, E2=E2, T=T))
        if metropolis(E1=E1, E2=E2, T=T):
            k += 1
            mc.generate_new_structure()
            shutil.copyfile('new_structure.data', 'NbMoTaW_{}.data'.format(k))
            print("第{}次交换".format(k) + str(atoms_pair))
