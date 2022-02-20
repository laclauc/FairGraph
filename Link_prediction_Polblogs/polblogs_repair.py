# Script by C. Laclau
# Last update 14/10

# Apply fair repairing on the graph structure with OT (EMD and Laplacian versions)

from src.util.link_prediction import *
from src.util.main_program import *
import pickle as pkl

print("Loading the graph")
_temp = nx.read_gml("polblogs/polblogs.gml")
g = _temp.to_undirected(reciprocal=False, as_view=False)
g.remove_nodes_from(list(nx.isolates(g)))


# Extract the attribute
node_list = list(g.nodes(data='value'))
s = convert(node_list)

nx.set_node_attributes(g, s, 's')

print("Repairing the graph with Laplace")

lap_adj, s, gamma, M = total_repair_reg(g, metric='euclidean', method="laplace", reg=1, case='bin', log=False,
                                        name='plot_cost_gamma')

# Convert the adjacency matrix back to a networkx type of graph
laplace_graph = [nx.from_numpy_matrix(lap_adj), s]

# Save the repaired graph in a pickle
with open('laplace_graph.pkl', 'wb') as outfile:
    pkl.dump(laplace_graph, outfile, pkl.HIGHEST_PROTOCOL)

print("Repairing the graph with EMD")

emd_adj, s, gamma, M = total_repair_emd(g, metric='euclidean', case='weighted', log=False, name='plot_cost_gamma')
emd_graph = [nx.from_numpy_matrix(emd_adj), s]

# Save the repaired graph in a pickle
with open('emd_graph.pkl', 'wb') as outfile:
    pkl.dump(emd_graph, outfile, pkl.HIGHEST_PROTOCOL)

