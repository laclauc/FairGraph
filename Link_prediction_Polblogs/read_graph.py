import pickle as pkl
import numpy as np
import networkx as nx

with open("laplace_graph_05.pkl", "rb") as f:
    mat = pkl.load(f)

graphe = mat[0]
S = mat[1]

print(np.size(adj))