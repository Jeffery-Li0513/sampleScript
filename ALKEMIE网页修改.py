import os
import re

# 修改源文件：
def alter(file, old_str, new_str):
    '''
    替换文件中需要修改的字符串，以只读方式打开文件，读取文件内容 --> 替换需要修改的行，然后再写入同名文件即可覆盖
    :param file: 需要修改的文件
    :param old_str: 需要替换的字符串
    :param new_str: 替换后的内容
    :return:
    '''
    lines = []                                              # 用于储存修改后的文件，以行存在列表中
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if old_str in line:
                # 如果要修改的字符串存在，就修改
                line = line.replace(old_str, new_str)
            lines.append(line)
    with open(file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

class switch_en_zhCN:
    def __init__(self, file, old_strs, modify_str, language=['en','zh_CN']):
        '''
        实现中英文切换
        :param file: 含有html文件的文件夹路径
        '''
        html_files = self.searchFile(file)
        self.modify(html_files, old_strs=old_strs, modify_str=modify_str, language=language)

    def searchFile(self, file):
        '''
        找到目录下所有的.html后缀文件
        :param file: 含有html文件的文件夹路径
        :return: 返回只含有html后缀的文件名
        '''
        html_files = []
        file_names = os.listdir(file)
        for file_name in file_names:
            if os.path.splitext(file_name)[1] == '.html':
                html_files.append(file_name)
        return html_files

    def modify(self, html_files, old_strs, modify_str, language=['en','zh_CN']):
        '''
        根据输入的需要替换的字符串进行修改，先找到指定的行，然后用replace替换路径
        :param html_files: 含有html文件的文件夹路径
        :param old_strs: 用来查找需要修改的行，包含整行
        :param modify_str: 指定行内需要修改的字符串，列表中第一个为中文需要修改的，第二个元素为英文需要修改的
        :param language: 指定语言
        :return:
        '''
        abs_html_files = []
        if language == 'zh_CN':
            for i in html_files:
                abs_html_files.append('source/zh_CN/build/html/' + i)
        elif language == 'en':
            for i in html_files:
                abs_html_files.append('source/en/build/html/' + i)

        for html_file in abs_html_files:                                       # 遍历所有html文件
            lines = []                                                         # 空列表用来储存修改后的文件，方便后面写入
            with open(html_file, 'r', encoding='utf-8') as f:                  # 打开文件
                for line in f:                                                 # 遍历文件的每一行
                    if language == 'zh_CN':
                        if old_strs[0] in line:
                            new_str = '../zh_CN/' + html_file.split('/')[-1]
                            line = line.replace(modify_str[0], new_str)
                        elif old_strs[1] in line:
                            new_str = '../en/' + html_file.strip('.html').split('/')[-1] + '_en.html'
                            line = line.replace(modify_str[1], new_str)
                    if language == 'en':
                        if old_strs[0] in line:
                            new_str = '../zh_CN/' + html_file.strip('_en.html').split('/')[-1] + '.html'
                            line = line.replace(modify_str[0], new_str)
                        elif old_strs[1] in line:
                            new_str = '../en/' + html_file.split('/')[-1]
                            line = line.replace(modify_str[1], new_str)
                    lines.append(line)
            with open(html_file, 'w', encoding='utf-8') as f:
                f.writelines(lines)

def expandContent():
    '''
    展开目录，主要就是修改index.html和index_en.html中的目录。只让前四个展开即可
    :return:
    '''
    files = ['source/zh_CN/build/html/index.html', 'source/en/build/html/index_en.html']
    # 检索的字符串：
    old_strs = ['''<li class="toctree-l1"><a class="reference internal" href="ALKEMIEIntroduction.html">简介</a></li>''',
                   '''<li class="toctree-l1"><a class="reference internal" href="ApplyForTrial.html">申请试用</a></li>''',
                   '''<li class="toctree-l1"><a class="reference internal" href="ALKEMIESoftwareConceptAndOverview.html">设计架构</a></li>''',
                   '''<li class="toctree-l1"><a class="reference internal" href="ALKEMIESoftwareInstallation.html">客户端安装</a></li>''',
                   '''<li class="toctree-l1"><a class="reference internal" href="ALKEMIEIntroduction_en.html">Introduction</a></li>''',
                   '''<li class="toctree-l1"><a class="reference internal" href="ApplyForTrial_en.html">Application</a></li>''',
                   '''<li class="toctree-l1"><a class="reference internal" href="ALKEMIESoftwareConceptAndOverview_en.html">Architecture</a></li>''',
                   '''<li class="toctree-l1"><a class="reference internal" href="ALKEMIESoftwareInstallation_en.html">Installation</a></li>''']
    # 替换的字符串：
    new_strs = ['''<li class="toctree-l1 current"><a class="current reference internal" href="ALKEMIEIntroduction.html">简介</a>
                  <ul>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIEIntroduction.html#id2">软件特色</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIEIntroduction.html#id3">软件功能</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIEIntroduction.html#id4">适用范围</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIEIntroduction.html#id5">开发人员</a></li>
                  </ul>
                </li>''',
                   '''<li class="toctree-l1 current"><a class="current reference internal" href="ApplyForTrial.html">申请试用</a>
                  <ul>
                    <li class="toctree-l2"><a class="reference internal" href="ApplyForTrial.html#id2">申请步骤</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="ApplyForTrial.html#id4">参与开发</a></li>
                  </ul>
                </li>''',
                   '''<li class="toctree-l1 current"><a class="current reference internal" href="ALKEMIESoftwareConceptAndOverview.html">设计架构</a>
                  <ul>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIESoftwareConceptAndOverview.html#alkemie-matter-studio">ALKEMIE Matter Studio
                        基础架构</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIESoftwareConceptAndOverview.html#id2">ALKEMIE Matter Studio 平台概述</a></li>
                  </ul>
                </li>''',
                   '''<li class="toctree-l1 current"><a class="current reference internal" href="ALKEMIESoftwareInstallation.html">客户端安装</a>
                  <ul>
                    <li class="toctree-l2"><a class="reference internal" href="CompilerEnvironment.html">编译环境</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="InstallationPython.html">安装python</a></li>
                    <li class="toctree-l2"><a class="reference internal"
                        href="InstallationVisualStudioInstaller.html">安装Visual Studio Installer</a></li>
                    <li class="toctree-l2"><a class="reference internal"
                        href="InstallationALKEMIESoftware.html">安装ALKEMIE客户端软件</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="SoftwareLogin.html">软件登录</a></li>
                    <li class="toctree-l2"><a class="reference internal"
                        href="CommonProblemsAndSolutions.html">常见问题及解决方法</a></li>
                  </ul>
                </li>''',
                   '''<li class="toctree-l1 current"><a class="current reference internal" href="ALKEMIEIntroduction_en.html">Introduction</a>
                  <ul>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIEIntroduction_en.html#characteristics">Characteristics</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIEIntroduction_en.html#function">Function</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIEIntroduction_en.html#scope-of-application">Scope of
                        Application</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="ALKEMIEIntroduction_en.html#developers">Developers</a></li>
                  </ul>
                </li>''',
                   '''<li class="toctree-l1 current"><a class="current reference internal" href="ApplyForTrial_en.html">Application</a>
                  <ul>
                    <li class="toctree-l2"><a class="reference internal" href="ApplyForTrial_en.html#step-of-application">Step of Application</a>
                    </li>
                    <li class="toctree-l2"><a class="reference internal" href="ApplyForTrial_en.html#become-a-developer">Become a Developer</a>
                    </li>
                  </ul>
                </li>''',
                   '''<li class="toctree-l1 current"><a class="current reference internal" href="ALKEMIESoftwareConceptAndOverview_en.html">Architecture</a>
                  <ul>
                    <li class="toctree-l2"><a class="reference internal"
                        href="ALKEMIESoftwareConceptAndOverview_en.html#the-overall-design-concept-of-alkemie-is-as-follows">The overall design concept of ALKEMIE is
                        as follows:</a></li>
                    <li class="toctree-l2"><a class="reference internal"
                        href="ALKEMIESoftwareConceptAndOverview_en.html#alkemie-matter-studio-platform-overview">ALKEMIE Matter Studio Platform Overview</a></li>
                  </ul>
                </li>''',
                   '''<li class="toctree-l1 current"><a class="current reference internal" href="ALKEMIESoftwareInstallation_en.html">Installation</a>
                  <ul>
                    <li class="toctree-l2"><a class="reference internal" href="CompilerEnvironment_en.html">Compiler
                        Environment</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="InstallationPython_en.html">Install
                        python</a></li>
                    <li class="toctree-l2"><a class="reference internal"
                        href="InstallationVisualStudioInstaller_en.html">Install Visual Studio Installer</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="InstallationALKEMIESoftware_en.html">Install
                        ALKEMIE software</a></li>
                    <li class="toctree-l2"><a class="reference internal" href="SoftwareLogin_en.html">Software login</a>
                    </li>
                    <li class="toctree-l2"><a class="reference internal" href="CommonProblemsAndSolutions_en.html">Common
                        Problems and Solutions</a></li>
                  </ul>
                </li>''']

    for file in files:
        for old_str in old_strs:
            new_str = new_strs[old_strs.index(old_str)]           # 找到对应要替换掉的字符串
            alter(file, old_str, new_str)

def hide_directory():
    '''
    修改ALKEMIEIntroduction.html，不显示里面的用户文档撰人员和联系人两个标题
    :return:
    '''
    file_name = ['source/zh_CN/build/html/ALKEMIEIntroduction.html','source/en/build/html/ALKEMIEIntroduction_en.html']
    # 检索字符串
    old_strs = ['<li class="toctree-l2"><a class="reference internal" href="#user-document">User Document</a></li>',
               '<li class="toctree-l2"><a class="reference internal" href="#contacts">Contacts</a></li>']
    new_strs = ['<!-- <li class="toctree-l2"><a class="reference internal" href="#user-document">User Document</a></li> -->',
                '<!-- <li class="toctree-l2"><a class="reference internal" href="#contacts">Contacts</a></li> -->']
    for file in file_name:
        for old_str in old_strs:
            new_str = new_strs[old_strs.index(old_str)]
            alter(file, old_str, new_str)

if __name__ == '__main__':
    # 实现中英文切换，把脚本放在source文件夹外层，所以路径就为：source/zh_CN/build/html/ ; source/en/build/html/
    path = ['source/zh_CN/build/html/', 'source/en/build/html/']
    old_strs = ['<a href="E:/桌面/convert/source6/zh_CN/build/html/*.html">中文</a>',
                '| <a href="E:/桌面/convert/source6/en/build/html/*_en.html">English</a>']
    modify_str = ['E:/桌面/convert/source6/zh_CN/build/html/*.html','E:/桌面/convert/source6/en/build/html/*_en.html']
    language = ['zh_CN', 'en']
    for i in range(2):
        item = switch_en_zhCN(path[i], old_strs, modify_str, language=language[i])

    expandContent()
    hide_directory()
    print("任务结束")