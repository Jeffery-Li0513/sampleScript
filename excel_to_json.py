import xlrd, json

def read_xlsx(path):
    # 打开Excel文件
    data = xlrd.open_workbook(path)
    # 读取第一个工作表
    table = data.sheets()[0]
    # 统计行数
    rows = table.nrows
    data = []               # 用于存放数据
    for i in range(1,rows):
        values = table.row_values(i)
        new_dict = {
            "number": int(values[0]),
            "title": values[1],
            "posttime": values[2],
            "author": values[3],
            "image": values[4]
        }
        for i in range(len(values)-5):
            p_name = 'p_' + str(i+1)
            if len(values[5+i]) != 0:
                new_dict[p_name]=values[5+i]
            elif len(values[5+i]) == 0:
                break
        data.append(new_dict)
    return data

if __name__ == '__main__':
    data = read_xlsx('News_list.xlsx')

    # Python对象转化为数组，sort_keys=False按原始顺序输出；indent=4为缩进；separators设置分隔符；ensure_ascii=False取消ASCII码输出
    json_data = json.dumps(data,sort_keys=False,indent=4,separators=(',',':'),ensure_ascii=False)

    jsFile = open('NewsData.json','w+',encoding='utf-8')
    jsFile.write(json_data)
    jsFile.close()