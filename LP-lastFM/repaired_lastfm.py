import csv, random, itertools
import networkx as nx
import numpy as np
# from src.util.link_prediction import *
from src.util.main_program import *
import pickle as pkl
import pandas as pd
from sklearn import preprocessing

with open('datasets/facebook_large/musae_facebook_edges.csv', newline='') as f:
	reader_2 = csv.reader(f)
	edges_data = list(reader_2)

del edges_data[0]

edges_data = [(int(i[0]),int(i[1])) for i in edges_data]

df = pd.read_csv('datasets/facebook_large/musae_facebook_target.csv')

user_id = list(df['id'])
page_type = list(df['page_type'])

# ['company', 'government', 'politician', 'tvshow'] = [0, 1, 2, 3]
le = preprocessing.LabelEncoder()
le.fit(page_type)
# print(list(le.classes_))
page_type_new = le.transform(page_type)

G = nx.from_edgelist(edges_data)
node_list = list(G.nodes())

protS = dict([(i,j) for i,j in zip(user_id,page_type_new)])# dictionary of node is as key and sens-attr as value

# attribute in order of nodes of G
sens_attr = [protS[i] for i in node_list]



from src.util.main_program import *
import numpy as np
import pickle as pkl


def free_support_barycenter_laplace(measures_locations, measures_weights, X_init, reg_type='disp', reg_laplace=1e-1, reg_source=1,
                                    b=None, weights=None, numItermax=1, stopThr=1e-4, verbose=False, log=None, metric = 'sqeuclidean'):
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

    while (displacement_square_norm > stopThr and iter_count < numItermax):

        T_sum = np.zeros((k, d))

        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights, weights.tolist()):

            M_i = ot.dist(X, measure_locations_i, metric=metric)
            T_i = ot.da.emd_laplace(xs=X, xt=measure_locations_i, a=b, b=measure_weights_i, M=M_i,
                                    reg=reg_type, eta=reg_laplace, alpha=reg_source, verbose=verbose, numItermax=10)

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


protS_array = sens_attr
X0 = []

X = nx.to_numpy_array(G)
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

X_bary, log = free_support_barycenter_laplace(measures_locations=Xs, measures_weights=weights, reg_type='disp', reg_laplace=1,
                                              reg_source=1, X_init=X_init, b=b, log=True, metric='sqeuclidean', verbose=True)
couplings = log['T']
X0.append(X_bary)

for i in range(len(classes)):
    X_repaired.append(np.dot(couplings[i].T, X_bary))

X0.append(np.concatenate(X_repaired))
new_adj = np.concatenate(X_repaired)

new_g = nx.from_numpy_matrix(new_adj)


repaired_graph = [new_g, protS]


with open('fb_laplace1_graph.pkl', 'wb') as outfile:
    pkl.dump(repaired_graph, outfile, pkl.HIGHEST_PROTOCOL)

node_id = list(g.nodes)
node_list = list(g.nodes(data='value'))
lab_node_list = [(node_list.index(i), i[0]) for i in node_list]
lab_node_array = np.array(lab_node_list)

list_edge = [(u, v) for (u, v, d) in new_g.edges(data=True) if d['weight'] <= 2e-4]
new_g.remove_edges_from(list_edge)
lab = {k: j for k, j in zip(new_g.nodes, lab_node_array[:, 1])}
h = nx.relabel_nodes(new_g, lab)

repaired_graph = [h, protS]

print(nx.density(G))
print(nx.density(new_g))
print(nx.density(h))

with open('fb_laplace1_graph_lessdense.pkl', 'wb') as outfile:
    pkl.dump(repaired_graph, outfile, pkl.HIGHEST_PROTOCOL)



