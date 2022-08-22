'''


'''


# 将两个列表合并为一个字典，有三种方法
keys_list = ['A', 'B', 'C']
values_list = ['blue', 'red', 'bold']
# 方法一
dict_method_1 = dict(zip(keys_list, values_list))
# 方法二
dict_method_2 = {keys_list:value for key,value in zip(keys_list, values_list)}
# 方法三
items_tuples = zip(keys_list, values_list)
dict_method_3 = {}
for key, value in items_tuples:
    if key in dict_method_3:
        pass
    else:
        dict_method_3[key] = value


# 将两个或多个列表合并为一个包含列表的列表
def merge(*args, missing_val=None):
    '''
    :param args: 传入的多个列表
    :param missing_val: 比较短的列表后面的填充值
    :return: 返回合并好的大列表
    '''
    max_length = max([len(lst) for lst in args])
    outList = []
    for i in range(max_length):
        outList.append([args[k][i] if i < len(args[k]) else missing_val for k in range(len(args))])
    return outList


# 对字典列表进行排序
dicts_lists = [
    {
        "Name": "James",
        "Age": 20,
    },
    {
        "Name": "May",
        "Age": 14,
    },
    {
        "Name": "Katy",
        "Age": 23,
    },
]
dicts_lists.sort(key=lambda item: item.get("Age"))              # 用sort函数，不过值为数字
from operator import itemgetter
f = itemgetter("Name")
dicts_lists.sort(key=f)