'''
该脚本用来批量修改alkemine网页版的HTML文件的中英文切换问题
主要是通过修改HTML中class="rst-other-versions"内部的url来实现
要改成相对路径
'''

import os
import re

zh_CN_filename_list = os.listdir(r'E:\桌面\source6\zh_CN_html')
en_filename_list = os.listdir(r'E:\桌面\source6\en_html')
# print(len(zh_CN_filename_list))
# print(len(en_filename_list))
# for i in zh_CN_filename_list:
#     print(i + '    ' + en_filename_list[zh_CN_filename_list.index(i)])


# 修改中文版中的
os.chdir(r'E:\桌面\source6\zh_CN_html')
for filename in zh_CN_filename_list:
    new_filename = 'E:/桌面/source6/zh_CN_1/' + filename
    file_split = filename.split('.')
    with open(filename,'r',encoding='utf-8') as f1, open(new_filename,'w',encoding='utf-8') as f2:
    # with open(filename, 'r', encoding='utf-8') as f1:
        file = f1.readlines()
        for item in file:
            if (item.strip() == '<a href="E:/桌面/convert/source6/zh_CN/build/html/*.html">中文</a>'):
                # print(item)
                new_name = '../zh_CN/' + filename
                item = item.replace('E:/桌面/convert/source6/zh_CN/build/html/*.html',new_name)
            elif (item.strip() == '| <a href="E:/桌面/convert/source6/en/build/html/*_en.html">English</a>'):
                # print(item)
                new_name = '../en/' + en_filename_list[zh_CN_filename_list.index(filename)]
                item = item.replace('E:/桌面/convert/source6/en/build/html/*_en.html',new_name)
            f2.write(item)


# 修改英文版的
os.chdir(r'E:\桌面\source6\en_html')
for filename in en_filename_list:
    new_filename = 'E:/桌面/source6/en_1/' + filename
    with open(filename,'r',encoding='utf-8') as f1, open(new_filename,'w',encoding='utf-8') as f2:
        file = f1.readlines()
        for item in file:
            if (item.strip() == '<a href="E:/桌面/convert/source6/zh_CN/build/html/*.html">中文</a>'):
                new_name = '../zh_CN/' + zh_CN_filename_list[en_filename_list.index(filename)]
                item = item.replace('E:/桌面/convert/source6/zh_CN/build/html/*.html',new_name)
            elif (item.strip() == '| <a href="E:/桌面/convert/source6/en/build/html/*_en.html">English</a>'):
                new_name = '../en/' + filename
                item = item.replace('E:/桌面/convert/source6/en/build/html/*_en.html',new_name)
            f2.write(item)


'''
有点问题：index文件和IndexStructureAndVisualizationIn180000DatabasesThroughFinder文件会混在一起
'''