from src.util.main_program import *
import numpy as np
import pickle as pkl
import facebook


# load the network
facebook.load_network()
adj = nx.adjacency_matrix(facebook.network)

# Look at a node's features
j = 0
gender = []
protSnum = np.empty(4039)
protSnum2 = np.empty(4039)
list_node = np.asarray(facebook.network.nodes) # order of nodes
list_node_t = map(str, list_node)
protS = dict.fromkeys(list_node_t)

def Convert(tup):
    dic = dict(tup)
    return dic

for i in list_node:
    _temp = facebook.network.nodes[i]['features']
#    g = np.asarray[_temp[77], _temp[78]]
    if (_temp[77] == 1 and _temp[78] == 0) or _temp[77] == 2:
        protS[str(i)] = 1
        protSnum[i] = 1
        protSnum2[j] = 1
    else:
        protS[str(i)] = 0
        protSnum[i] = 0
        protSnum2[j] = 0
    j += 1

g = facebook.network
list_node_g = list((g.nodes()))
list_node_g = list(map(str, list_node_g))
list_node_g_dic = {i: list_node_g[i] for i in range(0, len(list_node_g))}

tups = tuple(zip(list_node_g, protSnum))
h = nx.relabel_nodes(g, list_node_g_dic)
dictionary = {}
protS = Convert(tups)

nx.set_node_attributes(h, protS, 's')
"""
print("Repairing the graph with EMD")

new_x_l, s, gamma, M = total_repair_emd(h, metric='euclidean', case='weighted', log=False, name='plot_cost_gamma')
new_g = nx.from_numpy_matrix(new_x_l)

emd_graph = [new_g, s]
with open('fb_emd_graph.pkl', 'wb') as outfile:
    pkl.dump(emd_graph, outfile, pkl.HIGHEST_PROTOCOL)
"""
print("Repairing the graph with Laplacian")

new_x_l, s, gamma, M = total_repair_reg(h, metric='euclidean', method="laplace", reg=5, case='bin', log=False,
                                    name='plot_cost_gamma')

new_g = nx.from_numpy_matrix(new_x_l)

laplace_graph = [new_g, s]
with open('fb_laplace_graph_5.pkl', 'wb') as outfile:
    pkl.dump(laplace_graph, outfile, pkl.HIGHEST_PROTOCOL)
