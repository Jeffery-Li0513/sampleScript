'''
处理gibbs2生成的.eos数据文件
'''

import os
import re

# 读取文件
def read_file(path):
    with open(path) as file:
        all_data = file.readlines()
        head_data = all_data[7].split()[1:]
        data = all_data[8:]
    split_data = []
    i = []
    for line in range(len(data)):
        try:
            if data[line] != '\n':
                strip_line = data[line].strip().split()
                strip_line = [float(x) for x in strip_line]
                i.append(strip_line)
            elif data[line] == '\n' and data[line+1] == '\n':
                split_data.append(i)
                i = []
            elif line == '\n' and data[data.index(line)+1] != '\n':
                continue
        except:
            continue
    return head_data, split_data

# 导出想要的数据，dat文件
def export_data_t(path, index, temp=None, pressure=None):
    '''
    01:p(GPa) 02:T(K) 03:V(bohr^3) 04:Estatic(Ha) 05:G(kJ/mol)
    06:Gerr(kJ/mol) 07:p_sta(GPa) 08:p_th(GPa) 09:B(GPa) 10:U-Esta(kJ/mol)
    11:Cv(J/molK) 12:F-Esta(kJ/mol) 13:S(J/molK) 14:ThetaD(K) 15:gamma
    16:alpha(10^-5/K) 17:dp/dT(GPa/K) 18:Bs(GPa) 19:Cp(J/molK) 20:B_Tp
    21:B_Tpp(GPa-1) 22:Fvib(kJ/mol) 23:Fel(kJ/mol) 24:Uvib(kJ/mol) 25:Uel(kJ/mol)
    26:Svib(J/molK) 27:Sel(J/molK) 28:Cv_vib(J/molK) 29:Cv_el(J/molK)
    :param index: 对应参数的索引
    :param path:
    :return:在目录下生成dat文件
    '''
    head_data, body_data = read_file(path)
    if temp == None and pressure != None:            # 没有指定温度时就是需要恒压下随温度的变化
        # name = os.path.basename(path).rstrip(".dat") + "-" + re.sub(u"\\(.*?\\)", "", head_data[index]) + ".dat"
        # 拼接文件名，源文件名去掉后缀 + index中的去掉括号 + .dat
        name = os.path.basename(path).rstrip(".dat") + "-" + re.sub(u"\\(.*?\\)", "", head_data[index]) + ".dat"
        with open(name, "w") as f:
            f.write(head_data[1] + "\t\t" + head_data[index] + "\n")
            for i in body_data:
                if i != []:
                    t = i[0][1]
                    k = i[0][index]
                    f.write(str(t) + "\t\t" + str(k) + "\n")


if __name__ == '__main__':
    # path = 'source_data/MoTa-0.8.eos'
    dirname = "source_data/"
    path_list = os.listdir(dirname)
    for path in path_list:
        export_data_t(path=(dirname + path), index=2, pressure=0.00)
