import numpy as np
import matplotlib.pyplot as plt
import ot
from scipy.sparse import issparse
import networkx as nx
from sklearn.manifold import TSNE
#from ot_laplacian import *
from node2vec import Node2Vec
import os


def get_graph_prot(sizes=[150, 150], probs=[[0.15, 0.005], [0.005, 0.15]], number_class='binary', choice='random',
                   shuffle=0.2):
    """
     Generate a graph with a community structure, and where the nodes are assigned a protected attribute
    :param sizes:  number of nodes in each protected group
    :param probs: probabilities of links between the protected attribute, and within them
    :param number_class: the number of protected groups (binary or multi)
    :param choice: controls the dependency between the protected attribute and the community structure
         - random : the structure and the attribute are completely independent
         - partition : the structure and the attribute are dependent
    :param shuffle: when the choice is partition, it controls the degree of dependency (low value corresponding to stronger
    dependence.
    :return: the graph where the protected attribute is a feature of the nodes
    """

    # Generate a graph following the stochastic block model
    g = nx.stochastic_block_model(sizes, probs)

    # Check if the graph is connected
    if not nx.is_connected(g):
        print('Graph is not connected')
        g = nx.stochastic_block_model(sizes, probs)

    # Protected attribute
    n = np.sum(sizes)
    protS = np.zeros(n)
    g = np.asarray(probs).shape[0]
    p = np.ones(g)

    if choice == 'random':
        if number_class == 'multi':
            protS = np.random.choice(g, n, p=p*1/g)
        elif number_class == 'binary':
            protS = np.random.choice(2, n, p=p*1/2)

    elif choice == 'partition':
        part_idx = g.graph['partition']
        for i in range(len(part_idx)):
            protS[list(part_idx[i])] = i

        # Shuffle x% of the protected attributes to change the degree of dependence
        protS = shuffle_part(protS, prop_shuffle=shuffle)

        # Handle the case when S is binary but the partition >2
        if np.asarray(probs).shape[0] > 2 & number_class == 'binary':
            idx_mix = np.where(protS == 2)[0]
            _temp = np.random.choice([0, 1], size=(len(idx_mix),), p=[1./2, 1./2])
            i = 0
            for el in idx_mix:
                protS[el] = _temp[i]
                i += 1

    # Assign the attribute as a feature of the nodes directly in the graph
    dict_s = {i: protS[i] for i in range(0, len(protS))}
    nx.set_node_attributes(g, dict_s, 's')

    return g


def shuffle_part(protS, prop_shuffle=0.1):
    """
    Randomly shuffle some of the protected attributes
    :param protS: the vector to shuffle
    :param prop_shuffle: the proportion of label to shuffle
    :return: the shuffled vector
    """
    prop_shuffle = prop_shuffle
    ix = np.random.choice([True, False], size=protS.size, replace=True, p=[prop_shuffle, 1 - prop_shuffle])
    protS_shuffle = protS[ix]
    np.random.shuffle(protS_shuffle)
    protS[ix] = protS_shuffle
    return protS


def total_repair(g, metric='jaccard', case='weighted', algo='emd', reg=0, log=False, name='data'):
    """
    
    :param g:
    :param metric:
    :param case:
    :param algo:
    :param reg:
    :param log:
    :param name:
    :return:
    """

    otdists = ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
               'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
               'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule']

    if issparse(X):
        X = X.todense()

    #print(X[0][0])
    # Separate rows adjacency matrix based on the protected attribute
    idx_p0 = np.where(protS == 0)
    X_0 = X[idx_p0]

    idx_p1 = np.where(protS == 1)
    X_1 = X[idx_p1]

    # Get the barycenter between adj0 and adj1
    n0, d0 = X_0.shape
    n1, d1 = X_1.shape

    # Compute barycenters using POT library
    # Uniform distributions on samples
    a = np.ones((n0,))/n0
    b = np.ones((n1,))/n1

    # loss matrix
    if metric in otdists:
        M = np.asarray (ot.dist(X_0, X_1, metric=metric))
    elif metric == 'simrank':
        sim = nx.simrank_similarity(g)
        Msim = [[sim[u][v] for v in sorted(sim[u])] for u in sorted(sim)]
        M = np.asarray(Msim)
    M /= M.max()

    # Exact transport
    if algo == 'emd':
        gamma = ot.emd(a, b, M)
    elif algo == 'sinkhorn':
        gamma = ot.sinkhorn(a, b, M, reg)
    elif algo == 'laplacian':
        kwargs = {}
        kwargs['sim'] = 'gauss'
        kwargs['alpha'] = 0.5
        gamma = compute_transport(X_0, X_1, method='laplace', metric='jaccard', weights='unif', reg=1, solver=None, wparam=1, **kwargs)

    # Total data repair
    pi_0 = n0 / (n0+n1)
    pi_1 = 1 - pi_0

    X_0_rep = pi_0 * X_0 + n0 * pi_1 * np.dot(gamma, X_1)
    X_1_rep = pi_1 * X_1 + n1 * pi_0 * np.dot(gamma.T, X_0)

    new_X = np.zeros(X.shape)
    new_X[idx_p0, :] = X_0_rep
    new_X[idx_p1, :] = X_1_rep

    if case == 'bin':
        #threshold, upper, lower = 0.5, 1, 0
        #new_X = np.where(new_X >= threshold, upper, lower)
        new_X[np.where(new_X<np.quantile(new_X, 0.4))==0]

    if log:
        plt.imshow(gamma)
        plt.colorbar()
        plt.show()
        plt.savefig('gamma_' + name + '.png')

        plt.imshow(M)
        plt.colorbar()
        plt.show()
        plt.savefig('costMatrix_' + name + '.png')

    return new_X, gamma, M


def visuTSNE(X, protS, k=2, seed=0, plotName='tsne_visu'):

    # display TSNE scatter plot
    tsne = TSNE(n_components=k, random_state=seed)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(X)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    fig, ax = plt.subplots ()
    for g in np.unique(protS):
        i = np.where(protS == g)
        ax.scatter(x_coords[i], y_coords[i], label=g)
    ax.legend
    plt.savefig(plotName+'.png')
    plt.show()


def emb_matrix(g, S, dimension=32, walk_length=15, num_walks=100, window=10):

    node2vec = Node2Vec(g, dimensions=dimension, walk_length=walk_length, num_walks=num_walks)
    model = node2vec.fit(window=window, min_count=1)
    idx = list(map(int, model.wv.index2word))
    X = model.wv.vectors
    new_S = S[idx]

    return X, new_S

def load_graph(G,file_str,name): # REQUIRED FOR VERSE
    G_2 = G.copy()
    G_2.name = name
#     try:
#         with open(edgefile): pass
#     except:
    nx.write_edgelist(G, file_str,data=False)
    G_2.graph['edgelist'] = file_str
    G_2.graph['bcsr'] = './verse_input/' + G_2.name + '.bcsr'
    G_2.graph['verse.output'] = './verse_output/' + G_2.name + '.bin'
    try:
        with open(G_2.graph['bcsr']): pass
    except:
        os.system('python ../verse-master/python/convert.py ' + G_2.graph['edgelist'] + ' ' + G_2.graph['bcsr'])
    return G_2

def Verse(g,file_str,name):
    G = load_graph(g, file_str,name)
    orders = "../verse-master/src/verse -input " + G.graph['bcsr'] + " -output " + G.graph['verse.output']+ " -dim 32"+" -alpha 0.85"
    os.system(orders)
    verse_embeddings = np.fromfile(G.graph['verse.output'],np.float32).reshape(g.number_of_nodes(), 32)

    return verse_embeddings

def read_emb(file_to_read):
    #read embedding file where first line is number of nodes, dimension of embedding and next lines are node_id, embedding vector
    with open(file_to_read, 'r') as f:
        number_of_nodes, dimension = f.readline().split()
        number_of_nodes = int(number_of_nodes)
        dimension = int(dimension)
        Y = [[0 for i in range(dimension)] for j in range(number_of_nodes)]
        for i, line in enumerate(f):
            line = line.split()
            Y[int(line[0])] = [float(line[j]) for j in range(1, dimension+1)]
    return Y

def fairwalk(input_edgelist,output_emb_file,dict_file):
    #compute node2vec embedding
    orders = 'python ./fairwalk/src/main.py'+' --input '+input_edgelist+' --output '+'./fairwalk/emb/'+output_emb_file+' --sensitive_attr '+dict_file+' --dimension 32'
    os.system(orders)
    #print('DONE!')
    embeddings = read_emb('./fairwalk/emb/'+output_emb_file)
    return embeddings

def getGraph_fairwalk(sizes=[150, 150], probs=[[0.15, 0.005], [0.005, 0.15]], seed=0, choice='random'):

    # Generate a graph
    g = nx.stochastic_block_model(sizes, probs, seed=seed)
    #print(g.nodes())
    adj_matrix = nx.adjacency_matrix(g)
    if not nx.is_connected(g):
        print('Graph is not connected')
    # Protected attribute
    n = np.sum(sizes)
    protS = np.zeros(n)
    np.random.seed(seed)
    if choice == 'random':
        protS = np.random.choice([0, 1], size=(n,), p=[1./2, 1./2])
    elif choice == 'partition':
        #dict_s = {i: protS[i] for i in range(0, len(protS))}
        #nx.set_node_attributes(g, dict_s, 's')
        part_idx = g.graph['partition']
        for i in range(len(part_idx)):
            protS[list(part_idx[i])] = i
        # Shuffle 20% of the protected attributes
        protS = shuffle_part(protS, prop_shuffle=0.2)
        if np.asarray(probs).shape[0] > 2:
            idx_mix = np.where(protS == 2)[0]
            _temp = np.random.choice([0, 1], size=(len (idx_mix),), p=[1./2, 1./2])
            i = 0
            for el in idx_mix:
                protS[el] = _temp[i]
                i += 1
    dict_s = {i: protS[i] for i in range(0, len(protS))}
    return g, adj_matrix, protS, dict_s