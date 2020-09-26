from scipy.sparse import csr_matrix, save_npz, load_npz, vstack, hstack, lil_matrix
import numpy as np
import pickle as pkl
from OTAdjacency import *
import sys
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

# Open the adjacency matrix and the protected attribute
datapath = sys.argv[1]
outputpath = sys.argv[2]

inputML = pkl.load(open(datapath, 'rb'))
X = inputML[0]
protS = inputML[1]

print("Construction graph Knn")
user_sim = kneighbors_graph(X, 40, mode='connectivity', include_self=True)

g = nx.from_scipy_sparse_matrix(user_sim, create_using=nx.Graph())

# Correct the graph
new_adj, gamma, M = total_repair(user_sim, protS, metric='euclidean', case='weighted', algo='emd', reg=1, log=False)

print(new_adj.shape)
ml1m_repaired_sinkhorn = [new_adj, protS]

with open(outputpath+'.pkl', 'wb') as outfile:
    pkl.dump(ml1m_repaired_sinkhorn, outfile, pkl.HIGHEST_PROTOCOL)



