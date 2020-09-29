from src.util.link_prediction import *
from src.util.main_program import *
import pickle as pkl
import numpy as np

print("Loading the graph")
_temp = nx.read_gml("polblogs/polblogs.gml")
g = _temp.to_undirected(reciprocal=False, as_view=False)
g.remove_nodes_from(list(nx.isolates(g)))
node_list = list(g.nodes(data='value'))

int_node_list = [(node_list.index(i), i[1]) for i in node_list]
protS_int = convert(int_node_list)

# Attribute as a dictionary
tups = node_list
s = convert(tups)
s_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

nx.set_node_attributes(g, s, 's')

# print("Repairing the graph with Laplace")

# new_x_l, s, gamma, M = total_repair_reg(g, metric='euclidean', method="laplace", reg=0.05, case='bin', log=False,
     #                                name='plot_cost_gamma')
# new_g = nx.from_numpy_matrix(new_x_l)

# laplace_graph = [new_g, s]
#with open('laplace_graph_005.pkl', 'wb') as outfile:
#    pkl.dump(laplace_graph, outfile, pkl.HIGHEST_PROTOCOL)


# print("Repairing the graph with EMD")

# new_x_l, s, gamma, M = total_repair_emd(g, metric='euclidean', case='weighted', log=False, name='plot_cost_gamma')
# new_g = nx.from_numpy_matrix(new_x_l)

# emd_graph = [new_g, s]
# with open('emd_graph.pkl', 'wb') as outfile:
#    pkl.dump(emd_graph, outfile, pkl.HIGHEST_PROTOCOL)

print("Adding random edges to the graph")
new_g = repair_random(g, s_arr, prob=0.005)
random_graph = [new_g, s]
with open('repPolblogs/random_graph_.pkl', 'wb') as outfile:
    pkl.dump(random_graph, outfile, pkl.HIGHEST_PROTOCOL)

# [0.01, 0.05, 0.1, 0.15, 0.2]
