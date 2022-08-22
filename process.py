'''
要求：接受传入的数据并解析，数据是一个列表，列表中包含节点指向关系
列表中每一个元素可以是元组 [(node1, node2), (node2, node3)]

'''

from graphviz import Digraph
from graphviz import Source
from graphviz import render

def draw(edges):
    p = Digraph(filename='test.gv')

    for edge in edges:
        p.edge(edge[0], edge[1])

    p.render(view=True)

edges = [('a','b'), ('a','c'), ('b','d'), ('d','e')]

# draw(edges)

# 渲染已经存在的dot文件
# src = render('dot', 'png', 'subgraph.dot')

# 设置子图指向子图
from graphviz import Digraph

g = Digraph('G', filename='cluster_edge.gv')
g.attr(compound='true')

with g.subgraph(name='cluster0') as c:
    c.edges(['ab', 'ac', 'bd', 'cd'])

with g.subgraph(name='cluster1') as c:
    c.edges(['eg', 'ef'])

g.edge('b', 'f', lhead='cluster1')
g.edge('d', 'e')
g.edge('c', 'g', ltail='cluster0', lhead='cluster1')
g.edge('c', 'e', ltail='cluster0')
g.edge('d', 'h')

g.view()