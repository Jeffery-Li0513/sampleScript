from graphviz import Digraph
from graphviz import render

'''
传入的数据结构：
    元组中的元素都是串联的，相互之间有依赖关系
    列表中的元素都是并联的，相互之间没有依赖关系

    并联的节点用圆形的
    串联的节点用方框
'''
nodes = ((['incar', 'kpoint', 'potcar'], 'input', 'relax'),
         (['incar', 'kpoint', 'potcar'], 'input', 'static'),
         (['incar', 'kpoint', 'dos'], 'input', 'band'),
         [(['incar', 'kpoint', 'potcar'], 'input', 'relax'),
          (['incar', 'kpoint', 'dos'], 'input', 'band'),(['incar', 'kpoint', 'dos'], 'input', 'band')])

# 只要是元组就创建一个子图，这样比较方便
TUPLE_NUMBER = 0
def tuple_subgraph(parent_graph, tuple):
    global TUPLE_NUMBER, subgraph_nodes
    cluster_name = 'cluster_tuple' + str(TUPLE_NUMBER)
    TUPLE_NUMBER += 1
    # 接收元组中节点的列表
    tuple_nodes = []
    with parent_graph.subgraph(name=cluster_name) as A:
        A.attr(style='', compound='true')
        # 创建节点
        for i in tuple:
            # 定义一个节点的名称索引，为了防止重复，在名称前面加上其所属子图名
            node_name = cluster_name + '_' + str(tuple.index(i))
            if type(i) is list:
                # 调用列表型的子图
                B, list_nodes = list_subgraph(A, i)
                tuple_nodes.append(list_nodes)
            elif type(i) is str:
                A.node(node_name, label=i)
                # a = (node_name, cluster_name)
                a = (node_name, A.name)
                tuple_nodes.append(a)
            elif type(i) is tuple:
                n = tuple_subgraph(A, i)
                tuple_nodes.append(n)
        # 连接节点
        for i in range(len(tuple)-1):
            # 连接元组内部列表和普通节点
            if type(tuple[i]) is list:
                index = (len(tuple[i]) // 2)
                # A.edge(list_nodes[index][0], tuple_nodes[i+1][0], ltail=list_nodes[index][0])
                A.edge(list_nodes[index][0], tuple_nodes[i + 1][0], ltail=B.name)
            if type(tuple[i]) is str:
                A.edge(tuple_nodes[i][0], tuple_nodes[i+1][0])
    # subgraph_nodes.append(tuple_nodes)
    # print(tuple_nodes)
    return tuple_nodes

# 如果是列表也创建一个子图，内部的节点是水平的
LIST_NUMBER = 0
def list_subgraph(parent_graph, l):
    global LIST_NUMBER, subgraph_nodes
    cluster_name = 'cluster_list' + str(LIST_NUMBER)
    LIST_NUMBER += 1
    # 暂时保存列表中的节点名
    list_nodes = []
    with parent_graph.subgraph(name=cluster_name) as B:
        B.attr(style='dashed', rank='same', compound='true')
        if have_tuple(l):
            n_list = []
            for i in l:
                n = tuple_subgraph(B, i)
                n_list.append(n)
            subgraph_nodes.append(n_list)
        else:
            for i in l:
                # 定义一个节点的名称索引，为了防止重复，在名称前面加上其所属子图名
                node_name = cluster_name + '_' + str(l.index(i))
                B.node(node_name, label=i)
                # a = (node_name, cluster_name)
                a = (node_name, B.name)
                list_nodes.append(a)
    # print(list_nodes)
    return B, list_nodes

# 查看列表中有没有元组，如果全部是字符串，则直接创建节点；如果含有元组，就调用元组函数
def have_tuple(l):
    for i in l:
        if type(i) is tuple:
            return True
    return False


# 全局图，连接子图的节点
p = Digraph(filename='test_process.gv')
p.attr(compound='true', size="15", center='true')

subgraph_nodes = []                     # 接受子图节点

for item in nodes:
    if type(item) is tuple:
        n = tuple_subgraph(p, item)
        subgraph_nodes.append(n)
    elif type(item) is list:
        list_subgraph(p, item)

# 根据返回的子图间的连接节点来连接子图，图的开头和结尾不用管
for item in range(len(nodes)-1):
    # 每个子图的最后一个元素与下个子图第一个元素连接
    if type(nodes[item]) is tuple:
        # 下一个元素不是列表才能直接连，否则得连多条线
        if type(nodes[item+1]) is not list:
            p.edge(subgraph_nodes[item][-1][0], subgraph_nodes[item+1][0][1][0], ltail=subgraph_nodes[item][-1][1], lhead=subgraph_nodes[item+1][0][1][1])
        elif type(nodes[item+1]) is list:
            for i in range(len(nodes[item+1])):
                p.edge(subgraph_nodes[item][-1][0], subgraph_nodes[item+1][i][0][1][0], ltail=subgraph_nodes[item][-1][1], lhead=subgraph_nodes[item+1][i][0][1][1], headport='n', tailport='s')
    # elif type(nodes[item]) is list:
    #     # 列表中每个子图的尾部都和下个元素子图的头部连接
    #     pass

p.view()
