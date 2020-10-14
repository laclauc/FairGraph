from src.util.main_program import *
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def free_support_barycenter_laplace(measures_locations, measures_weights, X_init, reg_type='disp', reg_laplace=1e-1, reg_source=1,
                                    b=None, weights=None, numItermax=100, stopThr=1e-7, verbose=False, log=None, metric = 'sqeuclidean'):
    """
    Solves the free support (locations of the barycenters are optimized, not the weights) Wasserstein barycenter problem (i.e. the weighted Frechet mean for the 2-Wasserstein distance)

    The function solves the Wasserstein barycenter problem when the barycenter measure is constrained to be supported on k atoms.
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
            T_i = ot.da.emd_laplace(xs = X, xt = measure_locations_i, a = b, b = measure_weights_i, M=M_i,
                                    reg=reg_type, eta=reg_laplace, alpha=reg_source, numInnerItermax=200000, verbose=verbose)

            print('Done solving Laplace')

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

# Read author-author edge list and protected attribute
g = nx.read_edgelist("data/author-author.edgelist")
node_id = list(g.nodes)

# Read attribute
df = pd.read_csv("data/countries.attributes", delimiter=":", header=None)
df.columns = ["nodeId", "Country"]

# Convert nodeId to string to match the graph and encode the country
df["nodeId"] = df["nodeId"].astype(str)

LE = LabelEncoder()
df['Country'] = LE.fit_transform(df['Country'])

protS = dict(zip(df['nodeId'], df['Country']))

nx.set_node_attributes(g, protS, name='protS')


# Remove node in protS which are not in the graph (isolated nodes)
new = protS.copy()
for key, value in protS.items():
    if key not in node_id:
        new.pop(key)

protS = new.copy()
protS_array = np.array(list(protS.values()))

X0 = []

X = nx.to_numpy_array(g)
X0.append(X)
n, d = X.shape

classes = np.unique(protS_array)

idxs = []
Xs = []
X_repaired = []
weights = []
n_i = []
Cs = []

for i in classes:
    idxs.append(np.concatenate(np.where(protS_array == i)))
    n_i.append(len(idxs[-1]))
    b_i = np.random.uniform(0., 1., (n_i[-1],))
    b_i = b_i / np.sum(b_i)  # Dirac weights
    Xs.append(X[idxs[-1]])
    Ks = Xs[-1].dot(Xs[-1].T)
    Cs.append(Ks/Ks.max())
    weights.append(b_i)

X_init = np.random.normal(0., 1., X.shape)  # initial Dirac locations
b = np.ones((n,)) / n  # weights of the barycenter (it will not be optimized, only the locations are optimized)
# lambdast = np.ones((3,)) / 3

X_bary, log = free_support_barycenter_laplace(measures_locations=Xs, measures_weights = weights, reg_type='disp', reg_laplace=1,
                                              reg_source=1, X_init = X_init, b = b, log=True, metric='sqeuclidean', verbose=True)
couplings = log['T']
X0.append(X_bary)

for i in range(len(classes)):
    X_repaired.append(np.dot(couplings[i].T, X_bary))

X0.append(np.concatenate(X_repaired))
new_adj = np.concatenate(X_repaired)

new_g = nx.from_numpy_matrix(new_adj)

repaired_graph = [new_g, protS]
with open('dblp_laplace1_graph.pkl', 'wb') as outfile:
    pkl.dump(repaired_graph, outfile, pkl.HIGHEST_PROTOCOL)
