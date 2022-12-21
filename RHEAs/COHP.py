# -*- coding: utf-8 -*-
# @Time    : 2022/12/12 11:01
# @Author  : zefengLi
# @email   : by2201136@buaa.edu.cn
# @Comment : plot COHP by pymatgen


from pymatgen.electronic_structure.cohp import Cohp, get_integrated_cohp_in_energy_range
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Cohpcar, Icohplist
import re

COHP_path = "D:\Document\git\sampleScript\RHEAs\data\COHPCAR-2.0.lobster"
COHP_path_LIST = ["D:\Document\git\sampleScript\RHEAs\data\COHPCAR-2.0.lobster",
                  "D:\Document\git\sampleScript\RHEAs\data\COHPCAR-1.6.lobster",
                  "D:\Document\git\sampleScript\RHEAs\data\COHPCAR-1.2.lobster",
                  "D:\Document\git\sampleScript\RHEAs\data\COHPCAR-0.8.lobster",
                  "D:\Document\git\sampleScript\RHEAs\data\COHPCAR-0.0.lobster",
                  "D:\Document\git\sampleScript\RHEAs\data\COHPCAR+0.4.lobster",
                  "D:\Document\git\sampleScript\RHEAs\data\COHPCAR+0.8.lobster",
                  "D:\Document\git\sampleScript\RHEAs\data\COHPCAR+1.0.lobster"]
NAME_LIST = ["MoTa-2.0", "MoTa-1.6", "MoTa-1.2", "MoTa-0.8", "MoTa-0.0", "MoTa+0.4", "MoTa+0.8", "MoTa+1.0"]
COHPLIST_path = "D:\Document\git\sampleScript\RHEAs\data\ICOHPLIST.lobster"

# cohpcar = Cohpcar(filename=COHP_path)
# cdata = cohpcar.cohp_data
# cdata_processed = {}
# for key in cdata:
#     c = cdata[key]
#     c["efermi"] = 0
#     c["energies"] = cohpcar.energies
#     c["are_coops"] = False
#     cdata_processed[key] = Cohp.from_dict(c)

def average_all(filenames, namelist):
    cohpdata_processed = {}
    for i in filenames:
        cohpcar = Cohpcar(filename=i)
        cohpdata = cohpcar.cohp_data
        c = cohpdata['average']
        c["efermi"] = 0
        c["energies"] = cohpcar.energies
        c["are_coops"] = False
        cohpdata_processed[namelist[filenames.index(i)]] = Cohp.from_dict(c)

    return cohpdata_processed

cdata_processed = average_all(COHP_path_LIST, NAME_LIST)

cp = CohpPlotter()
cp.add_cohp_dict(cdata_processed)
# cp.add_cohp('average', cdata_processed['average'])
# cp.add_cohp('1', cdata_processed['1'])
x = cp.get_plot()
x.ylim([-10, 6])
x.show()


cohplist = Icohplist(filename=COHPLIST_path).icohpcollection
# print(cohplist.icohplist)
# print(cohplist.icohpcollection._list_atom1)

# 某种键
# bonds_processed = {}
# bond = ['Mo', 'Ta']
# for i in range(len(cohplist._list_atom1)):
#     atom_1 = re.sub(r'[0-9]+', '', cohplist._list_atom1[i])
#     atom_2 = re.sub(r'[0-9]+', '', cohplist._list_atom2[i])
#     if (atom_1 == bond[0] and atom_2 == bond[1]) or (atom_1 == bond[1] and atom_2 == bond[0]):
#         c = cdata[cohplist._list_labels[i]]
#         c["efermi"] = 0
#         c["energies"] = cohpcar.energies
#         c["are_coops"] = False
#         bonds_processed[cohplist._list_labels[i]] = Cohp.from_dict(c)



# print(cdata[cohplist._list_labels[1]].keys())
# print(cohplist.get_icohp_by_label('1'))
# print(list(bonds_processed.keys()))
# print(cohplist.get_summed_icohp_by_label_list(list(bonds_processed.keys())) / len(list(bonds_processed.keys())))
#
# cp = CohpPlotter()
# cp.add_cohp_dict(bonds_processed)
# x = cp.get_plot()
# x.ylim([-10, 6])
# x.show()