# 主要功能：读取文件名并修改，比如所有 文件名.txt 文件名_en.txt

'''
思路：1 先读取到文件名
     2 以 . 划分文件名字符串
     3 把添加的内容拼接到一起
'''

import os,re

filename_list = os.listdir(r'E:\桌面\convert\source6\en\local')   # os.listdir(path) 读取指定文件夹下的文件名，储存到列表中。
os.chdir(r'E:\桌面\convert\source6\en\local')                     # 把工作目录定位到指定目录
# new_filename_list = []

for filename in filename_list:
    filename_split = filename.split('.')                            # 将文件名以 . 分隔开
    new_filename = filename_split[0] + '_' + filename_split[1] + '.' + filename_split[2]

    os.rename(filename, new_filename)                               # 替换文件名
