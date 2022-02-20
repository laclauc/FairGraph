import networkx as nx
import numpy as np

def property_graph(graph='g1'):
    """
    Define the properties of the graph to generate
    :graph : type of desired graph. Options : ['g1','g2', 'g3', 'g4', 'g5']
    """
    if graph == 'g1':
        method = 'partition'
        sizes = [75, 75]
        probs = [[0.10, 0.005], [0.005, 0.10]]
        number_class = 'binary'

    elif graph == 'g2':
        method = 'random'
        sizes = [75, 75]
        probs = [[0.10, 0.005], [0.005, 0.10]]
        number_class = 'binary'

    elif graph == 'g3':
        method = 'partition'
        sizes = [125, 25]
        probs = [[0.15, 0.005], [0.005, 0.35]]
        number_class = 'binary'

    elif graph == 'g4':
        method = 'partition'
        probs = [[0.20, 0.002, 0.003], [0.002, 0.15, 0.003], [0.003, 0.003, 0.10]]
        sizes = [50, 50, 50]
        number_class = 'binary'

    elif graph == 'g5':
        method = 'partition'
        probs = [[0.20, 0.002, 0.003], [0.002, 0.15, 0.003], [0.003, 0.003, 0.10]]
        sizes = [50, 50, 50]
        number_class = "multi"

    return probs, sizes, number_class, method

def get_graph_prot(sizes=None, probs=None, number_class='binary',
                    choice='random', shuffle=0.1):
    """
     Generate a graph with a community structure, and where the nodes are
     assigned a protected attribute
    :param sizes:  number of nodes in each protected group
    :param probs: probabilities of links between the protected attribute,
     and within them
    :param number_class: the number of protected groups (binary or multi)
    :param choice: controls the dependency between the protected attribute and
    the community structure
         - random : the structure and the attribute are completely independent
         - partition : the structure and the attribute are dependent
    :param shuffle: when the choice is partition, it controls the degree of
    dependency (low value corresponding to
     stronger dependence.
    :return: the graph where the protected attribute is a feature of the nodes
    and a the attribute as a dictionary
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

        # Shuffle x% of the protected attributes
        prot_s = shuffle_part(prot_s, prop_shuffle=shuffle)

        # Handle the case when S is binary but the partition >2
        if (np.asarray(probs).shape[0] > 2) & (number_class == 'binary'):
            idx_mix = np.where(prot_s == 2)[0]
            _temp = np.random.choice([0, 1], size=(len(idx_mix),),
                        p=[1. / 2, 1. / 2])
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
    ix = np.random.choice([True, False], size=prot_s.size, replace=True,
                            p=[prop_shuffle, 1 - prop_shuffle])
    prot_s_shuffle = prot_s[ix]
    np.random.shuffle(prot_s_shuffle)
    prot_s[ix] = prot_s_shuffle
    return prot_s
