'''
将json结构文件转换为extxyz格式
'''


import json
import numpy as np
import os
import sys
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import extxyz, read
from ase import Atoms
from ase.io.db import read_json
from ase.io.vasp import read_vasp_out
import re


def json_to_extxyz(json_file, extxyz_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        print("{}:{}".format(data[0]['tags'], len(data)))
        for i in range(len(data)):
            # 先处理结构，然后处理力和能量
            structure = Structure.from_dict(data[i]['structure'], fmt='json')
            ase_structure = AseAtomsAdaptor.get_atoms(structure)                                # 将pymatgen的Structure对象转换为ase的Atoms对象
            ase_structure.arrays['forces'] = np.array(data[i]['outputs']['forces'])             # 将力添加到ase的Atoms对象中
            ase_structure.info['energy'] = np.array(data[i]['outputs']['energy'])               # 将能量添加到ase的Atoms对象中
            ase_structure.info['config_type'] = data[i]['tags']                          # 将config_type添加到ase的Atoms对象中
            extxyz.write_xyz(extxyz_file, ase_structure, append=True)
        # # 先处理结构，然后处理力和能量
        # structure = Structure.from_dict(data[0]['structure'], fmt='json')
        # ase_structure = AseAtomsAdaptor.get_atoms(structure)                                # 将pymatgen的Structure对象转换为ase的Atoms对象
        # ase_structure.arrays['forces'] = np.array(data[0]['outputs']['forces'])             # 将力添加到ase的Atoms对象中
        # ase_structure.info['energy'] = np.array(data[0]['outputs']['energy'])               # 将能量添加到ase的Atoms对象中
        # ase_structure.info['config_type'] = data[0]['tags']                          # 将config_type添加到ase的Atoms对象中
        # extxyz.write_xyz(extxyz_file, ase_structure, append=True)


if __name__ == '__main__':
    file_list = os.listdir('./')
    jsonfiles = [i for i in file_list if i.endswith('.json') and 'Test' not in i]
    for json_file in jsonfiles:
        json_to_extxyz(json_file, 'NbMoTaW_train.xyz')