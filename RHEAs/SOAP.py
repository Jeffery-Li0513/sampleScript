'''
构建SOAP描述符
'''

import numpy as np
import ase.io
from dscribe.descriptors import SOAP

# Read in the POSCAR file for the metal matrix
fname = 'POSCAR'
fhand = open(fname)
header = []
atoms = []
for n, line in enumerate(fhand):
    if n <=7:
        header.append(line)
    if n>7:
        line = line.rstrip().split()
        if len(line) ==3:
            atom = [float(line[0]), float(line[1]), float(line[2])]
            atoms.append(atom)
header[5] = 'H '+ header[5]
header[6] = str(1)+ ' '+header[6]


# Read in the H positions, embed the H atom into the metal matrix, and write it to a CONTCAR file
H_positions = []
for i in range(len(H_positions)):
    fname = 'CONTCAR'
    with open(fname, "w") as fwrite:
        for line in header:
            fwrite.writelines(line)
        atom = H_positions[i]
        line = []
        line.append(str(atom[0]) + '\t')
        line.append(str(atom[1]) + '\t')
        line.append(str(atom[2]) + '  '+ '\n')
        fwrite.writelines(line)
        for atom in atoms:
            line = []
            line.append(str(atom[0]) + '\t')
            line.append(str(atom[1]) + '\t')
            line.append(str(atom[2]) + ' '+'\n')
            fwrite.writelines(line)


# Use ase module to read in the CONTCAR file
model=ase.io.read('CONTCAR',format='vasp')
model.set_pbc([1,1,1])


# Parameter setting for the SOAP descriptors
rcut = 7
nmax = 4
lmax = 4
H_soap_desc = []
periodic_desc = SOAP(species=['H','Fe','Co','Ni','Cr','Mn'],rcut=rcut, \
                     average = 'off', nmax=nmax,lmax=lmax,periodic=True,sparse=False)
print()

# Create SOAP descriptor for the H atoms
H_soap = periodic_desc.create(model,positions =[0],n_jobs=-1)
H_soap_desc.append(H_soap[0])


