import re


i = 'Nb21Mo22Ta15W17'

elements= re.findall(r'[A-Za-z]{1,2}', i)        # 获取元素
fraction = re.findall(r'[0-9]{1,2}', i)         # 获取元素的含量

print(elements)
print(fraction)