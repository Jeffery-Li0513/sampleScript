'''
Constrain an atom index or a list of atom indices to move on a line only.
'''

from ase.constraints import FixedLine
from ase import Atoms
from ase.io import read, write


poscar = read("POSCAR", format="vasp")
c = FixedLine(
    a = [atom.index for atom in poscar],
    direction=[0, 0, 1],
)
poscar.set_constraint(c)
write('POSCAR_fixed', images=poscar, format="vasp")