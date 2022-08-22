'''
生成难熔高熵合金不同元素的组合，并写入excle中
'''

from itertools import combinations
import pandas as pd

elements = ['Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W']
binary_structures = []
ternary_structures = []
quaternary_structures = []
quinary_structures = []

binary = combinations(elements, 2)
ternary = combinations(elements, 3)
quaternary = combinations(elements, 4)
quinary = combinations(elements, 5)

for i in list(binary):
    binary_structures.append(''.join(i))
for i in list(ternary):
    ternary_structures.append(''.join(i))
for i in list(quaternary):
    quaternary_structures.append(''.join(i))
for i in list(quinary):
    quinary_structures.append(''.join(i))

binary_data = pd.DataFrame({'Alloys': binary_structures})
ternary_data = pd.DataFrame({'Alloys': ternary_structures})
quaternary_data = pd.DataFrame({'Alloys': quaternary_structures})
quinary_data = pd.DataFrame({'Alloys': quinary_structures})

with pd.ExcelWriter('structures.xlsx') as writer:
    binary_data.to_excel(writer, sheet_name='binary')
    ternary_data.to_excel(writer, sheet_name='ternary')
    quaternary_data.to_excel(writer, sheet_name='quaternary')
    quinary_data.to_excel(writer, sheet_name='quinary')


# ternary_data.to_csv('structures.csv', sep=',', index=True, header=True)
