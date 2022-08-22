# -*- coding:utf-8 _*-
__author__ = 'Kaiqi Li'
__revise__ = 'Kaiqi Li'
__time__ = '12/03/2020'
__email__ = 'baristali@buaa.edu.cn'

import numpy as np
import sys
import re
import pprint

def ipr_from_procar(Ef=None, Gamma=False):

    #把PROCAR每一行都储存在列表_procar里，然后另_info为第二行
    f = open('PROCAR', 'r')
    _procar = f.readlines()
    _info = _procar[1]
    #pprint.pprint(_info)

    # get number of Kpoint, bands and ions
    pattern = re.compile(r'\d+') # compile() 函数将一个字符串编译为字节代码。 # 查找数字
    # 把第二行的信息变成一个列表
    num = list(map(int, pattern.findall(_info))) # findall() 查找匹配正则表达式的字符串/map() 会根据提供的函数对指定序列做映射/使用 list() 转换为列表
    pprint.pprint(num)
    pattern2 = re.compile(r'\s+') # 匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]。
    circle_data = _procar[1:] #列表
    nlines_perband = 3 + 2 +num[2] #每条能带占的行数11
    nlines_perK = nlines_perband*num[1] + 3 #每个K点占的行数531

    Kpoints = []
    Energy = []
    if Gamma:
        num[0] = 1

    for i in range(num[0]): #在K点循环

        Kpoint = circle_data[nlines_perK * i: nlines_perK * (i + 1)][3:] # 该K点内的所有数据
        #pprint.pprint(Kpoint)
        Bands = []
        for j in range(num[1]):
            band = Kpoint[nlines_perband*j +4: nlines_perband*(j+1)]    # 获取该K点中第j条能带的数据，条数为原子数
            energy = float(re.split(pattern2, Kpoint[nlines_perband * j + 1].split('#')[1].strip())[1])

            Energy.append(energy)

            Ions = []
            for k in band:
                ion = k.strip().split('  ')[-1]
                #pprint.pprint(ion)
                Ions.append(ion)
                #pprint.pprint(Ions)
            Bands.append(Ions)
            #pprint.pprint(Bands)
        Kpoints.append(Bands)
        #pprint.pprint(Kpoints)
    nKpoints = np.array(Kpoints).astype('float64')
    pprint.pprint(nKpoints.shape)

    # 是否要把费米能级矫正为0
    if Ef:
        Energy = [i - Ef for i in Energy]

    IPR = []
    for i in range(num[0]):
        for j in range(num[1]):
            numerator = np.sum(np.square(nKpoints[i, j, :-1]))
            #pprint.pprint(numerator)
            denominator = np.sum(np.square(nKpoints[i, j, -1]))
            ipr = numerator/denominator
            IPR.append(ipr)

    xy = np.array([Energy, IPR], dtype='float64').T
    np.savetxt('ipr_gamma', xy, fmt='%.10f', delimiter=' ', newline='\n')
    # print(xy)


if __name__ == '__main__':
    try:
        Ef = float(sys.argv[1])
    except:
        Ef = None
    ipr_from_procar(Ef, Gamma=True)
