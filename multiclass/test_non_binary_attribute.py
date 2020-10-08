import pickle
from src.util.main_program import *
import networkx as nx

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

X_bary, log = ot.lp.free_support_barycenter(measures_locations=Xs, measures_weights = weights, X_init = X_init, b = b, log=True, metric='euclidean')

X_bary, log = free_support_barycenter_laplace(measures_locations=Xs, measures_weights = weights, reg_type='disp', reg_laplace=1e1,
                                              reg_source=1, X_init = X_init, b = b, log=True, metric='euclidean')
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

