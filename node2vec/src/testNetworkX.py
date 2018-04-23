import networkx as nx
import matplotlib.pyplot as plt

## add a node 
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])

H = nx.path_graph(10)
G.add_nodes_from(H)
G.add_node(H)
G.add_node('shopping')
## edges
G.add_edge(1, 2, {'weight': 3.1415})
e = (2, 3)
G.add_edge(*e)
G.add_edges_from([(1,2), (1,3)])
# G.add_edges_from(H.edges)
nx.draw(G, with_labels=True)
plt.show()
G.clear()


edgelist = [('n1','n2'), ('n1','n3'), ('n2','n3')]
H = nx.Graph(edgelist)
nx.draw(H, with_labels= True)
plt.show()
