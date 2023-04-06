from ase.io.vasp import read_vasp

# poscar_1 = read_vasp(file='POSCAR')
# print(poscar_1.positions[0])
# poscar_2 = read_vasp(file='CONTCAR')
# print(poscar_2.positions[0])
# print(poscar_1.cell)
# print(poscar_1.cell.lengths()[0])


# dic = {'a':1, 'c':3, 'b':2}
# a1 = sorted(dic.items(), key=lambda x: x[0])
# print(dict(a1))
# print(list(dic.keys()))

import copy
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

# fig, axes = plt.subplots(1, 1, figsize=(5, 10))
# fig.subplots_adjust(hspace=4)
#
# # print(axes[0])
#
# # fig = plt.figure()
# cmap1 = copy.copy(mpl.cm.viridis)
# norm1 = mpl.colors.Normalize(vmin=0, vmax=100)
# im1 = mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1)
# cbar1 = fig.colorbar(
#     im1, cax=axes, orientation='vertical',
#     ticks=np.linspace(0, 100, 11),
#     label='colorbar with Normalize'
# )
#
# plt.show()

dic = {'a':[]}
dic['a'].append(1)
print(dic)

import numpy as np
from numpy import array
a = np.array(-1.97729)
b = {('Nb', 'Nb'): [-1.4030699999999996], ('Mo', 'Nb'): [-1.4030699999999996], ('Nb', 'Ta'): [-1.4030699999999996], ('Nb', 'W'): [-1.4030699999999996], ('Mo', 'Mo'): [-1.4030699999999996], ('Mo', 'Ta'): [-1.4030699999999996], ('Mo', 'W'): [-1.4030699999999996], ('Ta', 'Ta'): [-1.4030699999999996], ('Ta', 'W'): [-1.4030699999999996], ('W', 'W'): [-1.4030699999999996]}
b[('Mo', 'Nb')].append(0.11545)
print(b)