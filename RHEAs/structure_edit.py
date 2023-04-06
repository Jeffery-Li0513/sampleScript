'''
1、根据产生的slab模型，构建真空层
2、生成层错模型
3、Constrain an atom index or a list of atom indices to move on a line only. 固定原子只有Z方向自由度
'''

from ase.io.vasp import read_vasp
from ase.build import add_vacuum
from ase.io import write, read
from ase.constraints import FixedLine
from subprocess import PIPE
import sys, os, subprocess
from ase.geometry import cell_to_cellpar
from shutil import copyfile

pbs_nodefile = sys.argv[1]
np = sys.argv[2]

# 构建真空层
vacuum_size = 10.0
def build_vacuum(path, vacuum_size):
    # atoms = read_vasp(file)
    atoms = read(path+"/POSCAR", format="vasp")
    add_vacuum(atoms, vacuum_size)
    write(filename=path+'/POSCAR-slab', images=atoms, format='vasp')
    copyfile(path+'/POSCAR-slab', path+'/POSCAR')


def fix_xy_plane(path):
    '''
    固定原子只沿z方向优化
    :param path:
    :return:
    '''
    poscar = read(path + "/POSCAR", format="vasp")
    c = FixedLine(
        a=[atom.index for atom in poscar],
        direction=[0, 0, 1],
    )
    poscar.set_constraint(c)
    write(path + '/POSCAR_fixed', images=poscar, format="vasp")
    copyfile(path + '/POSCAR_fixed', path + '/POSCAR')

def generate_stacking_falut(path, vacuum_size, move_array):
    '''
    产生层错模型
    :param path: 绝对路径
    :param vacuum_size: 真空层厚度
    :param move_array: 移动的方向矢量，[0.0, 1.14365625, 0.0]
    :return:
    '''
    poscar = read(path+"/POSCAR", format="vasp")
    cell_params = cell_to_cellpar(poscar.cell)
    z_length = cell_params[2]
    generate = "(echo n; echo POSCAR-move) | atomsk {} -select above {}*box Z -shift {} {} {} vasp && cp POSCAR-move POSCAR"\
        .format(path+"/POSCAR", 0.55*(z_length-vacuum_size)/z_length, move_array[0], move_array[1], move_array[2])
    p = subprocess.Popen(generate, shell=True, stdout=PIPE, cwd=path)
    p.wait()



if __name__ == '__main__':
    structure_file_path = "/home/zfli-ICME/caculation/HEAs/NbMoTaW/generate_structure/SAE-110/Mo-W/-2.1/best/"
    pwd = os.getcwd()
    run = f"mpirun -machinefile {pbs_nodefile} -np {np} vasp_std > output"
    for i in range(1,4):
        # 首先获得结构，构建真空层，然后优化
        POSCAR_file = structure_file_path + "best{}-POSCAR".format(i)
        command_1 = f"mkdir ./{i} && cp INCAR KPOINTS {POSCAR_file} ./{i} && cd ./{i} && cp best{i}-POSCAR POSCAR && " + \
            "sed -i '2c 1.0' POSCAR && vaspkit -task 103 && cd ../"
        p1 = subprocess.Popen(command_1, shell=True, cwd=pwd)
        p1.wait()
        build_vacuum(pwd + f"/{i}", vacuum_size)
        p2 = subprocess.Popen(run, shell=True, cwd=pwd+f"/{i}")
        p2.wait()

        # 对优化完的结构构建层错
        p = subprocess.Popen(f"mkdir -p ./{i}/relaxation", shell=True, cwd=pwd)
        p.wait()
        # 准备优化层错模型的输入文件
        p = subprocess.Popen(f"cd ./{i} && cp INCAR KPOINTS CONTCAR POTCAR ./relaxation && mv ./relaxation/CONTCAR ./relaxation/POSCAR && cd ../", shell=True, cwd=pwd)
        p.wait()

        move_array = [-0.80869, 0.57183, 0.0]  # 移动的方向矢量，[0.0, 1.14365625, 0.0]
        vacuum_size = 10.0
        generate_stacking_falut(path=pwd+f"/{i}/relaxation", vacuum_size=vacuum_size, move_array=move_array)
        fix_xy_plane(path=pwd+f"/{i}/relaxation")
        p2 = subprocess.Popen(run, shell=True, cwd=pwd + f"/{i}/relaxation")
        p2.wait()