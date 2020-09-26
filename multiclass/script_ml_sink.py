from scipy.sparse import csr_matrix, save_npz, load_npz, vstack, hstack, lil_matrix
import numpy as np
import pickle as pkl
from OTAdjacency import *
import sys

# Open the adjacency matrix and the protected attribute
datapath = sys.argv[1]
outputpath = sys.argv[2]
regularisation = [1e-2, 5e-1, 1e-1, 5e-1, 1, 1.5, 2, 5, 10]
inputML = pkl.load(open(datapath, 'rb'))
X = inputML[0]
protS = inputML[1]


for i in regularisation:
    # Correct the graph
    new_adj, gamma, M = total_repair(X, protS, metric='jaccard', case='weighted', algo='sinkhorn', reg=i, log=False)
    ml1m_repaired_sinkhorn = [new_adj, protS, i]
    with open(outputpath+'_'+str(i)+'.pkl', 'wb') as outfile:
        pkl.dump(ml1m_repaired_sinkhorn, outfile, pkl.HIGHEST_PROTOCOL)



