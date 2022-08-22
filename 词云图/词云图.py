import jieba
from wordcloud import WordCloud
from imageio import imread
import matplotlib.pyplot as plt

# 读取文本文件
path = "word.txt"
text = open(path, 'r', encoding="utf-8").read()      # 直接读取到变量

# 使用jieba库对文本进行分析
cut_text = ''.join(jieba.cut(text))
print(type(cut_text))
# 读取图片
color_mask = imread('heart.jpg')
# 生成词云
cloud = WordCloud(font_path='1574927118.ttf',   # 字体文件路径
                  background_color="white",     # 将白色设置为背景色，即非白色区域将填充词
                  mask=color_mask,
                  max_words=2000,               # 最大词语数，如果文件内容少的画，就填不满了
                  max_font_size=80)             # 最大词的大小
word_cloud = cloud.generate(cut_text)

# 输出图片
plt.axis('off')
plt.imshow(word_cloud)
plt.show()