'''

'''


import imageio
import requests
import demjson
import json
from bs4 import BeautifulSoup
import re



headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.42'
}

def get_paper_url(base_url):
    '''
    获取北航学报某一期中所有的文章链接
    :param base_url: 某一期所在的主页
    :return: 该期中所有的文章链接
    '''
    volume_issue = requests.get(base_url, headers=headers)
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


def get_image_url(url):
    '''
    获取一篇文章中的所有图片的链接
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
    title = re.sub('[: ]', '', title)
    # print(title.text)
    sentence = soup.find_all(attrs={'class': 'figure_img figure_type2'})  # 获取到了图片对应的标签
    for i in sentence:
        # print('https://bhxb.buaa.edu.cn' + i.attrs['onerror'].strip('this.onerror=null;this.src=').strip("'"))
        image_url.append('https://bhxb.buaa.edu.cn' + i.attrs['onerror'].strip('this.onerror=null;this.src=').strip("'"))  # 直接提取src中的内容不是对应的链接
    return image_url, title


# 读取连接中的图片并制作GIF图
def compose_gif(url_list, title):
    gif_images = []
    for url in url_list:
        # 通过读取url来获得图片，对应于北航学报一篇文章中的图片链接。
        gif_images.append(imageio.v2.imread(url))
    imageio.mimsave("./图片/{}.gif".format(title), gif_images, fps=2)


if __name__ == '__main__':
    base_url = "https://bhxb.buaa.edu.cn/bhzk/article/2022/9#c_3"  # 对应北航学报某一期的主页
    paper_url = get_paper_url(base_url)
    for url in paper_url:
        image_url, title = get_image_url(url)
        print(title)
        compose_gif(image_url, title)