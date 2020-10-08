import pickle
from src.util.main_program import *
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

graph = pickle.load(open("graph_ievgen.pkl", "rb"))

X0 = []
X = graph[0]
g = nx.from_scipy_sparse_matrix(X)

if issparse(X):
    X = X.todense()
    # np.fill_diagonal(X, 1)

X = np.squeeze(np.asarray(X))
X0.append(X)

n, d = X.shape
protS = graph[1]

# Predict the protected attributes
kfold = model_selection.KFold(n_splits=3, random_state=100, shuffle=True)
model_kfold = LogisticRegression(solver='lbfgs')

"""
print("Start learning embedding on the original graph")
embedding_origin, s_origin, mod = emb_node2vec(g, protS)

results_origin = model_selection.cross_val_score(model_kfold, embedding_origin, s_origin, cv=kfold,
                                                 scoring='f1_weighted')

print("AUC on the original graph is: %8.2f" % results_origin.mean())
"""

classes = np.unique(protS)

idxs = []
Xs = []
X_repaired = []
weights = []
n_i = []
Cs = []

for i in classes:
    idxs.append(np.concatenate(np.where(protS == i)))
    n_i.append(len(idxs[-1]))
    b_i = np.random.uniform(0., 1., (n_i[-1],))
    b_i = b_i / np.sum(b_i)  # Dirac weights
    Xs.append(X[idxs[-1]])
    Ks = Xs[-1].dot(Xs[-1].T)
    Cs.append(Ks/Ks.max())
    weights.append(b_i)

X_init = np.random.normal(0., 1., X.shape)  # initial Dirac locations
b = np.ones((n,)) / n  # weights of the barycenter (it will not be optimized, only the locations are optimized)
lambdast = np.ones((3,)) / 3

#X_bary, couplings = ot.gromov.gromov_barycenters(n, Cs, ps=weights, p=b, lambdas= lambdast, loss_fun = 'square_loss', max_iter=100, tol=1e-3)

#X_bary, log = ot.lp.free_support_barycenter(measures_locations=Xs, measures_weights = weights, X_init = X_init, b = b, log=True, metric='euclidean')

X_bary, log = free_support_barycenter_laplace(measures_locations=Xs, measures_weights = weights, reg_type='disp', reg_laplace=1e-1,
                                              reg_source=1, X_init = X_init, b = b, log=True, metric='sqeuclidean')
couplings = log['T']
X0.append(X_bary)

for i in range(len(classes)):
    X_repaired.append(np.dot(couplings[i].T, X_bary))

X0.append(np.concatenate(X_repaired))

plt.figure(figsize=(12, 9))
for i in range(len(X0)):
    plt.subplot(1, len(X0), i + 1)
    _temp = nx.from_numpy_matrix(X0[i])
    list_edge = [(u, v) for (u, v, d) in _temp.edges(data=True) if d['weight'] <= 0.001]
    _temp.remove_edges_from(list_edge)
    #pos = nx.kamada_kawai_layout(_temp)
    nx.draw(_temp, with_labels=True, node_size=100, node_color=protS)
plt.suptitle('Comparison between obtained graphs', fontsize=20)
plt.show()


new_adj = np.concatenate(X_repaired)
# new_adj = X_bary

new_g = nx.from_numpy_matrix(new_adj)

auc_origin = []
auc_repair = []


print("Start learning embedding on the repaired graph")
embedding_repair, s_repair, mod_rep = emb_node2vec(new_g, protS)

results_repair = model_selection.cross_val_score(model_kfold, embedding_repair, s_repair, cv=kfold,
                                                 scoring='f1_weighted')

print("AUC on the repaired graph is: %8.2f" % results_repair.mean())
auc_repair.append(results_repair.mean())
