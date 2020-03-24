from util.main_program import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

#from util.gae.preprocessing import mask_test_edges
#from matplotlib.lines import Line2D
#from matplotlib.patches import Circle

import sys

synthetic_case = sys.argv[1]
log = sys.argv[2]
trial = sys.argv[3]

if synthetic_case == 'g1':
    method = 'partition'
    sizes = [75, 75]
    probs = [[0.10, 0.005], [0.005, 0.10]]

elif synthetic_case == 'g2':
    method = 'random'
    sizes = [75, 75]
    probs = [[0.10, 0.005], [0.005, 0.10]]

elif synthetic_case == 'g3':
    method = 'partition'
    sizes = [125, 25]
    probs = [[0.15, 0.005], [0.005, 0.35]]

elif synthetic_case == 'g4':
    method = 'partition'
    probs = [[0.20, 0.002, 0.003], [0.002, 0.15, 0.003], [0.003, 0.003, 0.10]]
    sizes = [50, 50, 50]

auc_origin = []
auc_repair = []

for i in range(trial):
    # Generate the graph
    g, s = get_graph_prot(sizes=sizes, probs=probs, choice=method)

    # Correct the graph
    new_adj, gamma, M = total_repair_emd(g,  metric='euclidean', case='weighted', log=False)
    new_g = nx.from_numpy_matrix(new_adj)

    # Learn embedding
    embedding_origin, s_origin, idx_origin = emb_node2vec(g, s)
    embedding_repair, s_repair, idx_repair = emb_node2vec(new_g, s)

    # Predict the protected attributes
    kfold = model_selection.KFold(n_splits=3, random_state=100)
    model_kfold = LogisticRegression(solver='lbfgs')

    results_origin = model_selection.cross_val_score(model_kfold, embedding_origin, s_origin, cv=kfold,
                                                     scoring='roc_auc')
    auc_origin.append(results_origin.mean())

    results_repair = model_selection.cross_val_score(model_kfold, embedding_repair, s_repair, cv=kfold,
                                                     scoring='roc_auc')
    auc_repair.append(results_repair.mean())

    if log == 'True':
        prot0 = np.where(s == 0)[0]
        prot1 = np.where(s == 1)[0]

        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot0, node_color='steelblue', label='S = 0')
        nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot1, node_color='gold', label='S = 1')
        nx.draw_networkx_edges(g, pos=pos)
        plt.legend(loc="upper left", scatterpoints=1, prop={'size': 15})
        plt.tight_layout()

        plt.savefig(synthetic_case+'.eps', bbox_inches='tight', format='eps')
        plt.show()

all_results = [auc_origin, auc_repair]
with open('results/' + synthetic_case + '_graph_node2vec.pkl', 'wb') as outfile:
    pkl.dump(all_results, outfile, pkl.HIGHEST_PROTOCOL)
