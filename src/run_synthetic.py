from util.main_program import *
import networkx as nx
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
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

for i in range(trial):
    # Generate the graph
    g, s = get_graph_prot(sizes=sizes, probs=probs, choice=method)

    # Correct the graph
    new_adj, gamma, M = total_repair_emd(g,  metric='euclidean', case='weighted', log=False)
    new_g = nx.from_numpy_matrix(new_adj)

    
    if log == 'True':
        prot0 = np.where(s == 0)[0]
        prot1 = np.where(s == 1)[0]

        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot0, node_color='steelblue', label='S = 0')
        nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot1, node_color='gold', label='S = 1')
        nx.draw_networkx_edges(g, pos=pos)
        plt.legend(loc="upper left", scatterpoints=1, prop={'size': 15})
        plt.tight_layout()

        plt.savefig('g1.eps', bbox_inches='tight', format='eps')
        plt.show()


"""
dens_trial =[]

trial = 1
regularisation = [1e-2, 5e-1, 1e-1, 5e-1, 1, 1.5, 2, 5, 10]
for i in range(trial):
    #print('Starting trial : ' + trial)
    # Generate the graph
    g, adj, protS = getGraphAdj(sizes=sizes, probs=probs, choice=method)
    dens.append(nx.density(g))
    # Correct the graph

    new_adj, gamma, M = total_repair(adj, protS, metric='euclidean', case='weighted', algo='sinkhorn', reg=1e-2, log=False)
    new_g = nx.from_numpy_matrix(new_adj)

    # Node embedding on old graph
    X_old, protS_old = emb_matrix(g, protS)
    #visuTSNE(X_old, protS_old, k=2, seed=0, plotName='tsne_g1_old_emd')

    # Repair node embedding from old graph
    #new_X, ngamma, nM = total_repair(X_old, protS_old, metric='euclidean', case='weighted', algo='sinkhorn', reg=1e-1, log=False)
    #visuTSNE(new_X, protS_old, k=2, seed=0, plotName='tsne_g1_inter_emd')

    # Node embedding on repaired graph
    X_rep, protS_rep = emb_matrix(new_g, protS)
    #visuTSNE(X_rep, protS_rep, k=2, seed=0, plotName='tsne_g1_new_emd')

    # display TSNE scatter plot
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)

    Y = tsne.fit_transform(X_old)
    Z = tsne.fit_transform(X_rep)
    x1_coords = Y[:, 0]
    y1_coords = Y[:, 1]

    x2_coords = Z[:, 0]
    y2_coords = Z[:, 1]

    c = ['steelblue', 'gold']
    k = 0
    legend_elements = [Line2D([0], [0], marker='o', color='white', label='S = 0', markerfacecolor='steelblue', markersize=15),
                       Line2D([0], [0], marker='o', color='white', label='S = 1',  markerfacecolor='gold', markersize=15)]

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    for g in np.unique(protS_old):
        i = np.where(protS_old == g)
        ax[0].scatter(x1_coords[i], y1_coords[i], color=c[k], s=100)
        k += 1
    k = 0
    ax[0].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                     labelleft='off')
    for g in np.unique(protS_rep):
        i = np.where(protS_rep == g)
        ax[1].scatter(x2_coords[i], y2_coords[i], color=c[k], s=100)
        k += 1

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=2, fancybox=True, shadow=False, prop={'size': 30}, fontsize=30)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                     labelleft='off')
    plt.tight_layout(pad=5.0)
    plt.savefig('embbedding' + '.eps', bbox_inches='tight')
    plt.show()

    # Logistic regression to predict the protected attributes
    #kfold = model_selection.KFold(n_splits=5, random_state=100)
    #model_kfold = LogisticRegression(solver='lbfgs')

    #results_kfold = model_selection.cross_val_score(model_kfold, X_old, protS_old, cv=kfold, scoring='roc_auc')
    #auc_original.append(results_kfold.mean())

    # Logistic regression to predict the protected attributes from repaired graph
    #results_kfold_rep = model_selection.cross_val_score(model_kfold, X_rep, protS_rep, cv=kfold, scoring='roc_auc')
    #auc_rep.append(results_kfold_rep.mean())

    # Logistic regression to predict the protected attributes from repaired embedding
    #results_kfold_rep_emb = model_selection.cross_val_score(model_kfold, new_X, protS_old, cv=kfold, scoring='roc_auc')
    #auc_rep_emb.append(results_kfold_rep_emb.mean())
    #print(auc_rep)
    #print(auc_original)

set_results = [auc_original, auc_rep, auc_rep_emb]
with open('results/res_g1_repair_graph_node2vec_sinkhorn.pkl', 'wb') as outfile:
    pkl.dump(set_results, outfile, pkl.HIGHEST_PROTOCOL)

print(np.asarray(auc_original).mean())
print(np.asarray(auc_rep).mean())
print(np.asarray(auc_rep_emb).mean())

auc_rep = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_rep, protS_rep, test_size=0.3)

    # Node embedding with new graph
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)

    fpr, tpr, threshold = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    auc_rep.append(metrics.auc(fpr, tpr))
    print(auc_rep)
print(np.mean(auc_rep), np.std(auc_rep))
#cr = classification_report(y_test, clf.predict(X_test))
X0 = [adj.todense(), new_adj]
plt.figure(figsize=(12, 9))
plt.imshow(X0[0], cmap="Greys")
plt.axis('off')
plt.tight_layout()
plt.savefig('adjacency_matrix_3.eps', type="eps")
plt.show()

plt.figure(figsize=(12, 9))
plt.imshow(X0[1], cmap="Greys")
plt.axis('off')
plt.tight_layout()
plt.savefig('adjacency_matrix_rep_3.eps', type="eps")
plt.show()

plt.figure(figsize=(12, 9))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    _temp = nx.from_numpy_matrix(X0[i])
    #pos = nx.kamada_kawai_layout(_temp)
    nx.draw(_temp, with_labels=True, node_size=100, node_color=protS)
plt.suptitle('Comparison between obtained graphs', fontsize=20)
plt.show()

plt.figure(figsize=(8, 6))
_temp = nx.from_numpy_matrix(X0[0])
nx.draw(_temp, with_labels=False, node_size=80, edge_color='darkslategray', node_color=protS, cmap='summer')
plt.legend()
plt.savefig('graph_partition_2.png')
plt.show()


prot0 = np.where(protS == 0)[0]
prot1 = np.where(protS == 1)[0]

pos = nx.spring_layout(g)
nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot0, node_color='steelblue', label='S = 0')
nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot1, node_color='gold', label='S = 1')
nx.draw_networkx_edges(g, pos=pos)
plt.legend(scatterpoints=1, prop={'size': 15})
plt.savefig('graph_partition_imbalance.eps', format='eps')
plt.show()

plt.figure(figsize=(12, 9))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(X0[i])
plt.colorbar()
plt.suptitle('Comparison between adjacency matrices', fontsize=20)
plt.show()

set_results = [adj, new_adj, protS, node2vec_old, node2vec_rep]
with open('results/res_2_partition_eq.pkl', 'wb') as outfile:
    pkl.dump(set_results, outfile, pkl.HIGHEST_PROTOCOL)


# Link prediction
# On the embeddings
from scipy.spatial import distance_matrix



# Logistic regression to predict the protected attributes

X_train, X_test, y_train, y_test = train_test_split(X, protS, test_size=0.3)

# Node embedding with new graph
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)

roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
cr = classification_report(y_test, clf.predict(X_test))
print(roc)
print(cr)
#print(clf.score(X, protS))


"""
