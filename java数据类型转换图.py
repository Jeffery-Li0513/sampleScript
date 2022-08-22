from graphviz import Digraph

dot = Digraph(name="数据类型转换",filename="javaData",format="png")

dot.node('d','char')
dot.node('e','float')
dot.node('g','double')

dot.attr(rank='same')
dot.node('a','byte')
dot.node('b','short')
dot.node('c','int')
dot.node('f','long')

dot.edges(['ab','bc','cf','cg','eg','dc'])
dot.attr('edge',style='dashed')
dot.edges(['ce','fe','fg'])

# dot.source('''
#     digraph java{
#
#     }
# ''')

dot.render(view=True)