from src.util.link_prediction import *
import networkx as nx
from src.util.main_program import *
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec


print("Loading the graph")
_temp = nx.read_gml("polblogs/polblogs.gml")
g = _temp.to_undirected(reciprocal=False, as_view=False)
g.remove_nodes_from(list(nx.isolates(g)))
node_list = list(g.nodes(data='value'))

# Attribute as a dictionary
tups = node_list
dictionary = {}
s = convert(tups, dictionary)
s_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

print("Repairing the graph with the laplacian regularisation")
new_x_l, s, gamma, M = total_repair_reg(g, metric='euclidean', method="laplace", reg=1, case='bin', log=False,
                                       name='plot_cost_gamma')

data = [new_x_l, s]
with open('polblogs_rep_laplace_1.pkl', 'wb') as outfile:
    pkl.dump(data, outfile, pkl.HIGHEST_PROTOCOL)


print("Learning embedding")
new_g_l = nx.from_numpy_matrix(new_x_l)


emb_x, new_s = emb_node2vec(new_g_l, s, filename="")

