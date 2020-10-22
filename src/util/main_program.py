import matplotlib.pyplot as plt
import ot
from scipy.sparse import issparse
import networkx as nx
from sklearn.manifold import TSNE
from node2vec import Node2Vec
from util.ot_laplace_clean import *
import os
import numpy as np

def get_graph_prot(sizes=None, probs=None, number_class='binary', choice='random', shuffle=0.1):
    """
     Generate a graph with a community structure, and where the nodes are assigned a protected attribute
    :param sizes:  number of nodes in each protected group
    :param probs: probabilities of links between the protected attribute, and within them
    :param number_class: the number of protected groups (binary or multi)
    :param choice: controls the dependency between the protected attribute and the community structure
         - random : the structure and the attribute are completely independent
         - partition : the structure and the attribute are dependent
    :param shuffle: when the choice is partition, it controls the degree of dependency (low value corresponding to
     stronger dependence.
    :return: the graph where the protected attribute is a feature of the nodes and a the attribute as a dictionary
    """

    if sizes is None:
        sizes = [150, 150]

    if probs is None:
        probs = [[0.15, 0.005], [0.005, 0.15]]

    # Generate a graph following the stochastic block model
    g = nx.stochastic_block_model(sizes, probs)

    # Check if the graph is connected
    is_connected = nx.is_connected(g)
    while not is_connected:
        g = nx.stochastic_block_model(sizes, probs)
        is_connected = nx.is_connected(g)

    # Protected attribute
    n = np.sum(sizes)
    prot_s = np.zeros(n)
    k = np.asarray(probs).shape[0]
    p = np.ones(k)

    if choice == 'random':
        if number_class == 'multi':
            prot_s = np.random.choice(k, n, p=p * 1 / k)
        elif number_class == 'binary':
            prot_s = np.random.choice(2, n, p=p * 1 / 2)

    elif choice == 'partition':
        part_idx = g.graph['partition']
        for i in range(len(part_idx)):
            prot_s[list(part_idx[i])] = i

        # Shuffle x% of the protected attributes to change the degree of dependence
        prot_s = shuffle_part(prot_s, prop_shuffle=shuffle)

        # Handle the case when S is binary but the partition >2
        if (np.asarray(probs).shape[0] > 2) & (number_class == 'binary'):
            idx_mix = np.where(prot_s == 2)[0]
            _temp = np.random.choice([0, 1], size=(len(idx_mix),), p=[1. / 2, 1. / 2])
            i = 0
            for el in idx_mix:
                prot_s[el] = _temp[i]
                i += 1

    # Assign the attribute as a feature of the nodes directly in the graph
    dict_s = {i: prot_s[i] for i in range(0, len(prot_s))}
    nx.set_node_attributes(g, dict_s, 's')

    return g, dict_s


def shuffle_part(prot_s, prop_shuffle=0.1):
    """
    Randomly shuffle some of the protected attributes
    :param prot_s: the vector to shuffle
    :param prop_shuffle: the proportion of label to shuffle
    :return: the shuffled vector
    """
    prop_shuffle = prop_shuffle
    ix = np.random.choice([True, False], size=prot_s.size, replace=True, p=[prop_shuffle, 1 - prop_shuffle])
    prot_s_shuffle = prot_s[ix]
    np.random.shuffle(prot_s_shuffle)
    prot_s[ix] = prot_s_shuffle
    return prot_s


def repair_random(g, s, prob):
    """
    Repairing of the graph by adding random links between nodes of two different groups
    :param g: the graph
    :param s: protected attribute
    :return: the new graph
    """
    x = nx.adjacency_matrix(g)
    # s = nx.get_node_attributes(g, 's')
    # s_arr = np.fromiter(s.values(), dtype=int)
    s_arr = s
    # Separate rows adjacency matrix based on the protected attribute
    idx_p0 = np.where(s_arr == 0)[0]
    idx_p1 = np.where(s_arr == 1)[0]

    x_random = np.copy(x.todense())

    for i in idx_p0:
        for j in idx_p1:
            if x[i, j] == 0:
                x_random[i, j] = np.random.choice([0, 1], p=[1-prob, prob])
                x_random[j, i] = x_random[i, j]

    new_g = nx.from_numpy_matrix(x_random)
    nx.set_node_attributes(new_g, s, 's')
    print(nx.density(g))
    print(nx.density(new_g))

    return new_g


def total_repair_emd(g, metric='euclidean', case='weighted', log=False, name='plot_cost_gamma'):
    """
    Repairing of the graph with OT and the emd version
    :param g: a graph to repair. The protected attribute is a feature of the node
    :param metric: the distance metric for the cost matrix
    :param case: the new graph is by nature a weighed one. We can also binarize it according to a threshold ('bin')
    :param log: if true plot the cost matrix and the transportation plan
    :param name: name of the file to save the figures
    :return: the repaired graph, the transportation plan, the cost matrix
    """

    x = nx.adjacency_matrix(g)
    s = nx.get_node_attributes(g, 's')
    s = np.fromiter(s.values(), dtype=int)
    otdists = ['cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'mahalanobis', 'matching', 'seuclidean',
               'sqeuclidean', ]

    if issparse(x):
        x = x.todense()

    # Separate rows adjacency matrix based on the protected attribute
    idx_p0 = np.where(s == 0)
    x_0 = x[idx_p0]

    idx_p1 = np.where(s == 1)
    x_1 = x[idx_p1]

    # Get the barycenter between adj0 and adj1
    n0, d0 = x_0.shape
    n1, d1 = x_1.shape

    # Compute barycenters using POT library
    # Uniform distributions on samples
    a = np.ones((n0,)) / n0
    b = np.ones((n1,)) / n1

    # loss matrix
    if metric in otdists:
        m = np.asarray(ot.dist(x_0, x_1, metric=metric))
    elif metric == 'simrank':
        sim = nx.simrank_similarity(g)
        m_sim = [[sim[u][v] for v in sorted(sim[u])] for u in sorted(sim)]
        m = np.asarray(m_sim)
    m = np.asarray(m/m.max())

    # Exact transport

    gamma = ot.emd(a, b, m)

    # Total data repair
    pi_0 = n0 / (n0 + n1)
    pi_1 = 1 - pi_0

    x_0_rep = pi_0 * x_0 + n0 * pi_1 * np.dot(gamma, x_1)
    x_1_rep = pi_1 * x_1 + n1 * pi_0 * np.dot(gamma.T, x_0)

    new_x = np.zeros(x.shape)
    new_x[idx_p0, :] = x_0_rep
    new_x[idx_p1, :] = x_1_rep

    if case == 'bin':
        new_x[np.where(new_x < np.quantile(new_x, 0.4)) == 0]

    if log:
        plt.imshow(gamma)
        plt.colorbar()
        plt.show()
        plt.savefig('gamma_' + name + '.png')

        plt.imshow(m)
        plt.colorbar()
        plt.show()
        plt.savefig('costMatrix_' + name + '.png')

    return new_x, s, gamma, m


def total_repair_reg(g, metric='sqeuclidean', method="sinkhorn", reg=0.01, eta=1, case='bin', log=False, name='plot_cost_gamma'):
    """
    Repairing of the graph with OT and the sinkhorn version
    :param g: a graph to repair. The protected attribute is a feature of the node
    :param metric: the distance metric for the cost matrix
    :param method: xx
    :param reg : entropic regularisation term
    :param case: the new graph is by nature a weighed one. We can also binarize it according to a threshold ('bin')
    :param log: if true plot the cost matrix and the transportation plan
    :param name: name of the file to save the figures
    :return: the repaired graph, the transportation plan, the cost matrix
    """

    x = nx.adjacency_matrix(g)
    s = nx.get_node_attributes(g, 's')
    s = np.fromiter(s.values(), dtype=int)
    otdists = ['cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'mahalanobis', 'matching', 'seuclidean',
               'sqeuclidean', ]

    if issparse(x):
        x = x.todense()

    # Separate rows adjacency matrix based on the protected attribute
    idx_p0 = np.where(s == 0)
    x_0 = x[idx_p0]

    idx_p1 = np.where(s == 1)
    x_1 = x[idx_p1]

    # Get the barycenter between adj0 and adj1
    n0, d0 = x_0.shape
    n1, d1 = x_1.shape

    # Compute barycenters using POT library
    # Uniform distributions on samples
    a = np.ones((n0,)) / n0
    b = np.ones((n1,)) / n1

    # loss matrix
    if metric in otdists:
        m = np.asarray(ot.dist(x_0, x_1, metric=metric))
    elif metric == 'simrank':
        sim = nx.simrank_similarity(g)
        m_sim = [[sim[u][v] for v in sorted(sim[u])] for u in sorted(sim)]
        m = np.asarray(m_sim)
    m = np.asarray(m/m.max())

    # Sinkhorn transport
    if method == "sinkhorn":
        gamma = ot.sinkhorn(a, b, m, reg)

    elif method == 'laplace':
        # kwargs = {'sim': 'gauss', 'alpha': 0.5}
        kwargs = {'sim': 'knn', 'nn': 5, 'alpha': 0.5}
        gamma = compute_transport(x_0, x_1, method='laplace', metric=metric, weights='unif', reg=reg,
                                  nbitermax=5000, solver=None, wparam=1, **kwargs)
    elif method == 'laplace_sinkhorn':
        kwargs = {'sim': 'gauss', 'alpha': 0.5}
        # kwargs = {'sim': 'knn', 'nn': 3, 'alpha': 0.5}
        gamma = compute_transport(x_0, x_1, method='laplace_sinkhorn', metric=metric, weights='unif', reg=reg,
                                  nbitermax=3000, eta=eta, solver=None, wparam=1, **kwargs)

    elif method == 'laplace_traj':
        # kwargs = {'sim': 'gauss', 'alpha': 0.5}
        kwargs = {'sim': 'knn', 'nn': 3, 'alpha': 0.5}
        gamma = compute_transport(x_0, x_1, method='laplace_traj', metric=metric, weights='unif', reg=reg,
                                  nbitermax=3000, solver=None, wparam=1, **kwargs)

    # Total data repair
    pi_0 = n0 / (n0 + n1)
    pi_1 = 1 - pi_0

    x_0_rep = pi_0 * x_0 + n0 * pi_1 * np.dot(gamma, x_1)
    x_1_rep = pi_1 * x_1 + n1 * pi_0 * np.dot(gamma.T, x_0)

    new_x = np.zeros(x.shape)
    new_x[idx_p0, :] = x_0_rep
    new_x[idx_p1, :] = x_1_rep

    if log:
        plt.imshow(gamma)
        plt.colorbar()
        plt.show()
        plt.savefig('gamma_' + name + '.png')

        plt.imshow(m)
        plt.colorbar()
        plt.show()
        plt.savefig('costMatrix_' + name + '.png')

    return new_x, s, gamma, m


def visuTSNE(X, protS, k=2, seed=0, plotName='tsne_visu'):
    # display TSNE scatter plot
    tsne = TSNE(n_components=k, random_state=seed)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(X)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    fig, ax = plt.subplots()
    for g in np.unique(protS):
        i = np.where(protS == g)
        ax.scatter(x_coords[i], y_coords[i], label=g)
    ax.legend
    plt.savefig(plotName + '.png')
    plt.show()


def emb_node2vec(g, s, dimension=32, walk_length=15, num_walks=100, window=10, filename='node2vec'):
    """
    Compute the node embedding using Node2Vec
    :param g: a graph
    :param s: protected attribute (vector)
    :param dimension: dimension of the embedding
    :param walk_length: length of the random walk
    :param num_walks: number of walks
    :param window: window
    :param filename: name of the file containing the node2vec model
    :return: the embedding matrix and the associate protected attribute
    """

    node2vec = Node2Vec(g, dimensions=dimension, walk_length=walk_length, num_walks=num_walks)
    model = node2vec.fit(window=window, min_count=1)
    idx = list(map(int, model.wv.index2word))
    emb_x = model.wv.vectors
    new_s = s[idx]
    model.save(filename)
    return emb_x, new_s, model


def load_graph(g, file_str, name):
    """
    This function is required for Verse
    """

    g_2 = g.copy()
    g_2.name = name
    nx.write_edgelist(g, file_str, data=False)
    g_2.graph['edgelist'] = file_str
    g_2.graph['bcsr'] = './verse_input/' + g_2.name + '.bcsr'
    g_2.graph['verse.output'] = './verse_output/' + g_2.name + '.bin'
    try:
        with open(g_2.graph['bcsr']):
            pass
    except:
        os.system('python ../verse-master/python/convert.py ' + g_2.graph['edgelist'] + ' ' + g_2.graph['bcsr'])

    return g_2


def verse(g, file_str, name):
    g = load_graph(g, file_str, name)
    orders = "../verse-master/src/verse -input " + g.graph['bcsr'] + " -output " + g.graph['verse.output'] + \
             " -dim 32" + " -alpha 0.85"
    os.system(orders)
    verse_embeddings = np.fromfile(g.graph['verse.output'], np.float32).reshape(g.number_of_nodes(), 32)

    return verse_embeddings


def read_emb(file_to_read):
    # read embedding file where first line is number of nodes, dimension of embedding and next lines are node_id,
    # embedding vector

    with open(file_to_read, 'r') as f:
        number_of_nodes, dimension = f.readline().split()
        number_of_nodes = int(number_of_nodes)
        dimension = int(dimension)
        y = [[0 for i in range(dimension)] for j in range(number_of_nodes)]
        for i, line in enumerate(f):
            line = line.split()
            y[int(line[0])] = [float(line[j]) for j in range(1, dimension + 1)]
    return y


def fairwalk(input_edgelist, output_emb_file, dict_file):
    # compute node2vec embedding
    orders = 'python ./fairwalk/src/main.py' + ' --input ' + input_edgelist + ' --output ' + './fairwalk/emb/' \
             + output_emb_file + ' --sensitive_attr ' + dict_file + ' --dimension 32'
    os.system(orders)
    # print('DONE!')
    embeddings = read_emb('./fairwalk/emb/' + output_emb_file)
    return embeddings

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

    while (displacement_square_norm > stopThr and iter_count < numItermax):

        T_sum = np.zeros((k, d))

        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights,
                                                                       weights.tolist()):
            M_i = ot.dist(X, measure_locations_i, metric=metric)
            T_i = ot.da.emd_laplace(xs=X, xt=measure_locations_i, a=b, b=measure_weights_i, M=M_i,
                                     reg=reg_type, eta=reg_laplace, alpha=reg_source, numInnerItermax=200000,
                                     verbose=verbose, stopInnerThr=1e-7)
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
