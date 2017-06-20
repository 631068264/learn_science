#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/20 09:34
@annotation = ''
"""
import networkx as nx

G = nx.Graph()
G.add_nodes_from([1, 2, 3])
# [(1, {}), (2, {}), (3, {})]
print G.nodes()
G.add_edge(1, 2)
print G.edges()

# [(1, {'label': 'blue'}), (2, {}), (3, {})]
G.node[1]['label'] = 'blue'
print G.nodes(data=True)

# nx.draw(G)
# import matplotlib.pyplot as plt
#
# plt.show()

"""
Undirected graphs
G = nx.Graph()
Directed graphs
D = nx.DiGraph()


M = nx.MultiGraph()
MD = nx.MultiDiGraph()
"""

print G.neighbors(1)
print nx.degree_centrality(G)
