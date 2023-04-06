from ase.io import read
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor

atoms = read('XDATCAR.0')

structure = AseAtomsAdaptor.get_structure(atoms)
# print(structure.species)

# 计算指定原子的配位数
for i in range(len(structure)):
    try:
        target_atom_index = i  # 指定要计算配位数的原子的索引
        voronoi = VoronoiNN(cutoff=3.0)  # 初始化VoronoiNN对象
        coordination_number = voronoi.get_cn(structure, target_atom_index)

        print("原子{}的配位数为{}".format(structure[target_atom_index].specie.name, coordination_number))
    except:
        print("原子{}的配位数为0".format(structure[target_atom_index].specie.name))

