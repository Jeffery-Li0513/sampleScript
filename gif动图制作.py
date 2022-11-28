'''

'''


import imageio
import requests
import demjson
import json
from bs4 import BeautifulSoup
import re
import os
import urllib3

urllib3.disable_warnings()


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.56'
}

class Beihangxuebao:
    def __init__(self, baseurl):
        self.base_url = baseurl
        paper_url = self.get_paper_url()
        for url in paper_url:
            image_url, title = self.get_image_url(url)
            print(title)
            self.compose_gif(image_url, title)

    def get_paper_url(self):
        '''
        获取北航学报某一期中所有的文章链接
        :param base_url: 某一期所在的主页
        :return: 该期中所有的文章链接列表
        '''
        volume_issue = requests.get(self.base_url, headers=headers)
        volume_issue_html = volume_issue.content.decode('utf-8')
        volume_issue_soup = BeautifulSoup(volume_issue_html, 'html.parser')
        # 通过select定位到a标签
        title = volume_issue_soup.select('div[class="article-list"] > div[class="article-list-right"] > div[class="article-list-title"] > a')
        paper_url = []                  # 储存该期中所有的文章链接
        for i in range(len(title)):
            # 进行正则匹配，文章的连接中都有doi字样
            if 'doi' in title[i].attrs['href']:
                paper_url.append('https:' + title[i].attrs['href'])
                # print('https:' + title[i].attrs['href'])
        return paper_url

    def get_image_url(self, url):
        '''
        获取北航学报一篇文章中的所有图片的链接
        :param url: 一篇文章对应的链接
        :return: 返回图片列表和文章名称
        '''
        image_url = []
        res = requests.get(url, headers=headers)
        # new_res = demjson.encode(res.text, encoding='utf-8')        # 如果要用json格式的话，需要用demjson编码一下
        # print(json.loads(new_res))
        html = res.content.decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title').text
        title = re.sub('[: /<sub>]', '', title)
        # print(title.text)
        sentence = soup.find_all(attrs={'class': re.compile("figure_img figure_type[12]")}) # 获取到了图片对应的标签
        for i in sentence:
            # print('https://bhxb.buaa.edu.cn' + i.attrs['onerror'].strip('this.onerror=null;this.src=').strip("'"))
            image_url.append('https://bhxb.buaa.edu.cn' + i.attrs['onerror'].strip('this.onerror=null;this.src=').strip("'"))  # 直接提取src中的内容不是对应的链接
        return image_url, title

    # 读取连接中的图片并制作GIF图
    def compose_gif(self, url_list, title):
        gif_images = []
        for url in url_list:
            # 通过读取url来获得图片，对应于北航学报一篇文章中的图片链接。
            gif_images.append(imageio.v2.imread(url))
        if not os.path.exists('./图片/{}'.format(self.__class__.__name__)):
            os.makedirs('./图片/{}'.format(self.__class__.__name__))
        imageio.mimsave("./图片/{}/{}.gif".format(self.__class__.__name__, title), gif_images, fps=2)

class Fuhecailiaoxuebao:
    def __init__(self, baseurl):
        self.base_url = baseurl
        paper_url = self.get_paper_url()
        for url in paper_url:
            image_url, title = self.get_image_url(url)
            print(title)
            if len(image_url) > 0:
                self.compose_gif(image_url, title)
    def get_paper_url(self):
        '''
        获取复合材料学报某一期中所有的文章链接
        :param base_url: 某一期所在的主页
        :return: 该期中所有的文章链接列表
        '''
        volume_issue = requests.get(self.base_url, headers=headers)
        volume_issue_html = volume_issue.content.decode('utf-8')
        volume_issue_soup = BeautifulSoup(volume_issue_html, 'html.parser')
        # 通过select定位到a标签
        title = volume_issue_soup.select('div[class="article-list article-list-latest"] > \
        div[class="article-list-right"] > div[class="allwrap clearfix"] > div[class="listwrap fl"] \
        div[class="article-list-title"] > a')
        paper_url = []                  # 储存该期中所有的文章链接
        for i in range(len(title)):
            # 进行正则匹配，文章的连接中都有doi字样
            if 'doi' in title[i].attrs['href']:
                paper_url.append('https:' + title[i].attrs['href'])
                # print('https:' + title[i].attrs['href'])
        return paper_url
    def get_image_url(self, url):
        '''
        获取复合材料学报一篇文章中的所有图片的链接
        :param url: 一篇文章对应的链接
        :return: 返回图片列表和文章名称
        '''
        image_url = []
        res = requests.get(url, headers=headers)
        # new_res = demjson.encode(res.text, encoding='utf-8')        # 如果要用json格式的话，需要用demjson编码一下
        # print(json.loads(new_res))
        html = res.content.decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title').text
        title = re.sub('[: /<sub>]', '', title)
        # print(title.text)
        # sentence = soup.find_all(attrs={'class': 'figure_img figure_type1'})  # 获取到了图片对应的标签
        sentence = soup.find_all(attrs={'class': re.compile("figure_img figure_type[12]")})
        for i in sentence:
            # print('https://bhxb.buaa.edu.cn' + i.attrs['onerror'].strip('this.onerror=null;this.src=').strip("'"))
            image_url.append('https://fhclxb.buaa.edu.cn' + i.attrs['onerror'].strip('this.onerror=null;this.src=').strip("'"))  # 直接提取src中的内容不是对应的链接
        return image_url, title

    # 读取连接中的图片并制作GIF图
    def compose_gif(self, url_list, title):
        gif_images = []
        for url in url_list:
            # 通过读取url来获得图片，对应于北航学报一篇文章中的图片链接。
            gif_images.append(imageio.v2.imread(url))
        if not os.path.exists('./图片/{}'.format(self.__class__.__name__)):
            os.makedirs('./图片/{}'.format(self.__class__.__name__))
        imageio.mimsave("./图片/{}/{}.gif".format(self.__class__.__name__, title), gif_images, fps=2)

class Hangkongxuebao:
    def __init__(self, baseurl):
        self.base_url = baseurl
        paper_url = self.get_paper_url()
        for url in paper_url:
            image_url, title = self.get_image_url(url)
            print(title)
            print(image_url)
            if len(image_url) > 0:
                self.compose_gif(image_url, title)
    def get_paper_url(self):
        '''
        获取航空学报某一期中所有的文章链接
        :param base_url: 某一期所在的主页
        :return: 该期中所有的文章链接列表
        '''
        volume_issue = requests.get(self.base_url, headers=headers, verify=False)
        volume_issue_html = volume_issue.content.decode('utf-8')
        volume_issue_soup = BeautifulSoup(volume_issue_html, 'html.parser')
        # 通过select定位到a标签
        title = volume_issue_soup.select('div[class="index_check_cont"] > \
        div[class="index_tab_title clearfix"] > p[class="index_txt1 fl"] >  a')
        paper_url = []                  # 储存该期中所有的文章链接
        for i in range(len(title)):
            # 进行正则匹配，航空学报每期前几个链接都是abstrat，href中存在abstrat就不添加
            if 'abstract' not in title[i].attrs['href']:
                paper_url.append(title[i].attrs['href'])
                # print(title[i].attrs['href'])
        return paper_url
    def get_image_url(self, url):
        '''
        获取复合材料学报一篇文章中的所有图片的链接。因为传进来的文章链接指向的页面并没有图片，需要重新获取RichHTML按钮中的链接
        :param url: 一篇文章对应的链接
        :return: 返回图片列表和文章名称
        '''
        image_url = []
        # 获取ichHTML按钮中的链接
        res = requests.get(url, headers=headers, verify=False)
        html = res.content.decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        #
        # lsdy1('RICH_HTML','19334','https://hkxb.buaa.edu.cn','2022','article/2022/1000-6893/20221006.html');return false;
        richHTML_url = 'https://hkxb.buaa.edu.cn/' + re.findall(r"article/.*\.html", soup.find(attrs={'class': 'black-bg btn-menu'}).attrs['onclick'])[0]
        new_res = requests.get(richHTML_url, headers=headers, verify=False)
        richHTML = new_res.content.decode('utf-8')
        richHTML_soup = BeautifulSoup(richHTML, 'html.parser')
        title = richHTML_soup.find('title').text
        title = re.sub('[: /<sub>]', '', title)
        # print(title.text)
        # sentence = soup.find_all(attrs={'class': 'figure_img figure_type1'})  # 获取到了图片对应的标签
        sentence = richHTML_soup.find_all(attrs={'title': "点击查看原图"})
        for i in sentence:
            image_url.append('/'.join(richHTML_url.split('/')[:-1]) + '/' + i.attrs['src'])  # 直接提取src中的内容不是对应的链接
        return image_url, title

    # 读取连接中的图片并制作GIF图
    def compose_gif(self, url_list, title):
        gif_images = []
        for url in url_list:
            # 通过读取url来获得图片，对应于航空学报一篇文章中的图片链接。
            gif_images.append(imageio.imread(requests.get(url, headers=headers, verify=False).content))
        if not os.path.exists('./图片/{}'.format(self.__class__.__name__)):
            os.makedirs('./图片/{}'.format(self.__class__.__name__))
        imageio.mimsave("./图片/{}/{}.gif".format(self.__class__.__name__, title), gif_images, fps=2)

if __name__ == '__main__':
    # bhxb_base_url = "https://bhxb.buaa.edu.cn/bhzk/article/2022/9#c_3"  # 对应北航学报某一期的主页
    # Beihangxuebao(bhxb_base_url)
    # fhclxb_base_url = "https://fhclxb.buaa.edu.cn/article/2022/9"           # 复合材料学报某一期主页
    # Fuhecailiaoxuebao(fhclxb_base_url)
    # hkxb_base_uel = "https://hkxb.buaa.edu.cn/CN/volumn/volumn_1553.shtml"
    # Hangkongxuebao(hkxb_base_uel)
    # data = requests.get('https://hkxb.buaa.edu.cn/article/2022/1000-6893/PIC/hkxb-43-10-527481-1-1.jpg', headers=headers, verify=False).content
    # # with open('test.jpg', 'wb') as fb:
    # #     fb.write(data)
    # gif_images = imageio.read(data)
    # imageio.mimsave("test.gif", gif_images, fps=2)