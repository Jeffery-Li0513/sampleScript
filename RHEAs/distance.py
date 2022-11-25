from ase.io.vasp import read_vasp
from ase.io.xyz import read_xyz
from ase.geometry.analysis import Analysis
import os
import pandas as pd

'''
1、根据初始结构，得到目标原子对的索引
2、在每一帧中根据对应的索引计算原子对之间的距离
3、画出图形
'''

# poscar = read_vasp(file='data/best1-POSCAR')
#
# atom_pair = ['Mo', 'Ta']
# atom_pair_indexs = []                   # (0, 70), (0, 71), (0, 92)······
#
# # neigh_list = neighbor_list(quantities='j', a=poscar, cutoff=2.8)
# # element_index = []
# # for i in atom_pair:
# #     element_index.append([atom.index for atom in poscar if atom.symbol == i])
# # element_index_dict = dict(zip(atom_pair, element_index))
# #
#
#
# # # 获取原子对索引
# # for atom_i_index in element_index_dict[atom_pair[0]]:
# #     for atom_j_index in element_index_dict[atom_pair[1]]:
# #         if abs(poscar.get_distance(a0=atom_i_index, a1=atom_j_index, mic=True) - first_neighbors) <= 0.01:
# #             atom_pair_indexs.append([atom_i_index, atom_j_index])
# #
# # print(element_index_dict)
# # print(atom_pair_indexs)
# # print(len(atom_pair_indexs))
#
#
# first_neighbors = sorted(poscar.get_all_distances()[0])[1]
#
# ana = Analysis(poscar)
# for i in ana.get_bonds(atom_pair[0], atom_pair[1])[0]:
#     if abs(poscar.get_distance(a0=i[0], a1=i[1], mic=True) - first_neighbors) <= 0.01:
#         atom_pair_indexs.append(i)


def cacu_atom_pair_indexs(atoms, atom_pair):
    '''
    计算对应键的索引
    :param atoms: ASE的atoms对象
    :param atom_pair: 需要计算的原子对，列表类型
    :return:
    '''
    atom_pair_indexs = []  # (0, 70), (0, 71), (0, 92)······
    first_neighbors = sorted(atoms.get_all_distances()[0])[1]
    ana = Analysis(atoms)
    for i in ana.get_bonds(atom_pair[0], atom_pair[1])[0]:
        if abs(atoms.get_distance(a0=i[0], a1=i[1], mic=True) - first_neighbors) <= 0.01:
            atom_pair_indexs.append(i)
    return atom_pair_indexs


def average_bond_length(atoms, atom_pair_indexs):
    '''
    计算一个结构中对应的键长，并取平均
    :param atoms: ASE的atoms对象
    :param atom_pair_indexs: 键索引
    :return:平均键长
    '''
    bond_length = []
    for i in atom_pair_indexs:
        length = atoms.get_distance(a0=i[0], a1=i[1], mic=True)
        bond_length.append(length)
    # print(atoms.get_distance(a0=32, a1=1, mic=True))
    return sum(bond_length) / len(bond_length)


if __name__ == '__main__':
    atom_pairs = [['Mo', 'Ta'], ['Mo', 'W'], ['Mo', 'Nb'], ['Ta', 'W'], ['Ta', 'Nb'], ['W', 'Nb']]
    atom_pairs_bond_dict = {}
    for atom_pair in atom_pairs:
        atoms = read_vasp(file='data/best1-POSCAR')
        atom_pair_indexs = cacu_atom_pair_indexs(atoms, atom_pair)
        bond_length_list = []
        contcar_list = sorted(os.listdir('data/'))
        contcar_list.remove('best1-POSCAR')
        for contcar in contcar_list:
            atoms = read_vasp(file='data/' + contcar)
            bond_length = average_bond_length(atoms, atom_pair_indexs)
            bond_length_list.append(round(bond_length, 3))
        atom_pairs_bond_dict['-'.join(atom_pair)] = bond_length_list
        print(bond_length_list)
    print(atom_pairs_bond_dict)
    # 输出
    df = pd.DataFrame(atom_pairs_bond_dict)
    print(df)
    df.to_csv('MoTa+0.8.txt', sep='\t', index=True)