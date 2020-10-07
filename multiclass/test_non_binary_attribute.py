import pickle
from src.util.main_program import *
import networkx as nx

def free_support_barycenter_laplace(measures_locations, measures_weights, X_init, b=None, reg_laplace =1e-1, weights=None, numItermax=100, stopThr=1e-7, verbose=False, log=None, metric = 'sqeuclidean'):
    """
    Solves the free support (locations of the barycenters are optimized, not the weights) Wasserstein barycenter problem with Laplacian regularization (i.e. the weighted Frechet mean for the 2-Wasserstein distance)

    The function solves the Wasserstein barycenter problem with Laplacian regularization when the barycenter measure is constrained to be supported on k atoms.
    This problem is considered in [1] (Algorithm 2). There are two differences with the following codes:
    - we do not optimize over the weights
    - we do not do line search for the locations updates, we use i.e. theta = 1 in [1] (Algorithm 2). This can be seen as a discrete implementation of the fixed-point algorithm of [2] proposed in the continuous setting.

    Parameters
    ----------
    measures_locations : list of (k_i,d) numpy.ndarray
        The discrete support of a measure supported on k_i locations of a d-dimensional space (k_i can be different for each element of the list)
    measures_weights : list of (k_i,) numpy.ndarray
        Numpy arrays where each numpy array has k_i non-negatives values summing to one representing the weights of each discrete input measure

    X_init : (k,d) np.ndarray
        Initialization of the support locations (on k atoms) of the barycenter
    b : (k,) np.ndarray
        Initialization of the weights of the barycenter (non-negatives, sum to 1)
    weights : (k,) np.ndarray
        Initialization of the coefficients of the barycenter (non-negatives, sum to 1)

    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    X : (k,d) np.ndarray
        Support locations (on k atoms) of the barycenter

    References
    ----------

    .. [1] Cuturi, Marco, and Arnaud Doucet. "Fast computation of Wasserstein barycenters." International Conference on Machine Learning. 2014.

    .. [2]  Álvarez-Esteban, Pedro C., et al. "A fixed-point approach to barycenters in Wasserstein space." Journal of Mathematical Analysis and Applications 441.2 (2016): 744-762.

    """

    iter_count = 0

    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = np.ones((k,))/k
    if weights is None:
        weights = np.ones((N,)) / N

    X = X_init

    log_dict = {}
    displacement_square_norms = []
    Ti = []

    displacement_square_norm = stopThr + 1.

    while ( displacement_square_norm > stopThr and iter_count < numItermax ):

        T_sum = np.zeros((k, d))

        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights, weights.tolist()):

            M_i = ot.dist(X, measure_locations_i, metric = metric)
            T_i = ot.da.emd_laplace(a = b, b = measure_weights_i, xs = X, xt = measure_locations_i, M = M_i,
                                    sim='knn', sim_param=None, reg='pos', eta=reg_laplace, alpha=.5,
                                    numItermax=100, stopThr=1e-9, numInnerItermax=100000,
                                    stopInnerThr=1e-9, log=False, verbose=False)
            T_sum = T_sum + weight_i * np.reshape(1. / b, (-1, 1)) * np.matmul(T_i, measure_locations_i)
            Ti.append(T_i)
        displacement_square_norm = np.sum(np.square(T_sum-X))
        if log:
            displacement_square_norms.append(displacement_square_norm)

        X = T_sum

        if verbose:
            print('iteration %d, displacement_square_norm=%f\n', iter_count, displacement_square_norm)

        iter_count += 1

    if log:
        log_dict['displacement_square_norms'] = displacement_square_norms
        log_dict['T'] = Ti[-N:]
        return X, log_dict
    else:
        return X


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

X_bary, log = free_support_barycenter_laplace(measures_locations=Xs, measures_weights = weights, reg_laplace=1e-1, X_init = X_init, b = b, log=True)
couplings = log['T']
X0.append(X_bary)

for i in range(len(classes)):
    X_repaired.append(np.dot(couplings[i].T, X_bary))

X0.append(np.concatenate(X_repaired))

plt.figure(figsize=(12, 9))
for i in range(len(X0)):
    plt.subplot(1, len(X0), i + 1)
    _temp = nx.from_numpy_matrix(X0[i])
    #pos = nx.kamada_kawai_layout(_temp)
    nx.draw(_temp, with_labels=True, node_size=100, node_color=protS)
plt.suptitle('Comparison between obtained graphs', fontsize=20)
plt.show()


new_adj = np.concatenate(X_repaired)
# new_adj = X_bary

new_g = nx.from_numpy_matrix(new_adj)

"""
print("Before embedding")

model = LogisticRegressionCV(cv=5, multi_class='multinomial', max_iter=1000, random_state=100).fit(X, protS)
print(model.score(X, protS))

model = LogisticRegressionCV(cv=5, multi_class='multinomial', max_iter=1000, random_state=100).fit(new_adj, protS)
print(model.score(new_adj, protS))

# Node embedding on old graph
X_old, protS_old = emb_matrix(g, protS)

# Repair node embedding from old graph
#new_X, ngamma, nM = total_repair(X_old, protS_old, metric='cosine', case='weighted', algo='laplacian', reg=2e-1, log=False)

# Node embedding on repaired graph
X_rep, protS_rep = emb_matrix(new_g, protS)

#visuTSNE(X_old, protS_old, k=2, seed=0, plotName='tsne_g1_old')
#visuTSNE(X_rep, protS_rep, k=2, seed=42, plotName='tsne_g1_new')

print("After embedding")

# Logistic regression to predict the protected attributes
#model = LogisticRegressionCV(cv=5, , scoring = 'roc_auc').fit(X_old, protS_old)
model = LogisticRegressionCV(cv=5, multi_class = 'multinomial', max_iter=1000, random_state=100).fit(X_old, protS_old)
print(model.score(X_old, protS_old))

model = LogisticRegressionCV(cv=5, multi_class = 'multinomial', max_iter=1000, random_state=100).fit(X_rep, protS_rep)
print(model.score(X_rep, protS_rep))


"""
