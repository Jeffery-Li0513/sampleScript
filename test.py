import re

print(re.sub('[:/]', '', '碳/碳复合材料的激光烧蚀行为与机制'))
print(re.sub('[:/<sub>]', '', 'BN<sub>mf<sub>-Si<sub>3<sub>N<sub>4w<sub>Si<sub>3<sub>N<sub>4<sub>复合材料的制备与性能'))
print(re.findall(r'^.*/[0-9]*\.html$', 'https://hkxb.buaa.edu.cn/article/2022/1000-6893/20221028.html'))
print('https://hkxb.buaa.edu.cn/article/2022/1000-6893/20221028.html'.rstrip(r'[0-9]\.html'))

print('/'.join('https://hkxb.buaa.edu.cn/article/2022/1000-6893/20221028.html'.split('/')[:-1]))