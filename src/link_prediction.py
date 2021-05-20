import numpy as np


def convert(tup):
    di = dict(tup)
    return di

def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0

def hadamard(model, data, link_info, protS_int):
    hadamard_links = []
    for i in range(len(data)):
        had_pro = np.multiply(model.wv[str(data[i][0])], model.wv[str(data[i][1])])
        absolute_diff = abs(protS_int[data[i][0]]-protS_int[data[i][1]])
        hadamard_links.append((had_pro, absolute_diff, link_info[i]))
    return hadamard_links


def hadamard_verse(model, data, link_info, protS_int):
    hadamard_links = []
    for i in range(len(data)):
        had_pro = np.multiply(model.wv[str(data[i][0])], model.wv[str(data[i][1])])
        absolute_diff = abs(protS_int[data[i][0]]-protS_int[data[i][1]])
        hadamard_links.append((had_pro, absolute_diff, link_info[i]))
    return hadamard_links


def hadamard_fb(model, data, link_info, protS_int):
    hadamard_links = []
    for i in range(len(data)):
        had_pro = np.multiply(model.wv[str(data[i][0])], model.wv[str(data[i][1])])  # hadamard product
        absolute_diff = abs(protS_int[str(data[i][0])]-protS_int[str(data[i][1])])
        hadamard_links.append((had_pro, absolute_diff, link_info[i]))
    return hadamard_links


def get_tups_data(hadamard_data):
    vectors_and_abs_val = []
    links = []
    for i in hadamard_data:
        vectors_and_abs_val.append((i[0], i[1]))
        links.append(i[2])
    return vectors_and_abs_val, links


def transform_str_to_int(orig_node_list, edges):
    """node_list is the tuple list in order of graph"""
    for i in range(len(edges)):
        ind_first_ele = [y[0] for y in orig_node_list].index(edges[i][0])
        ind_sec_ele = [y[0] for y in orig_node_list].index(edges[i][1])
        edges[i] = (ind_first_ele, ind_sec_ele)
    return edges


# V1
def splitGraphToTrainTest(un_graph, train_ratio, is_undirected=True):
    # taken and adapter from GEM repository of palash1992
    train_graph = un_graph.copy()
    test_graph = un_graph.copy()

    for (st, ed, w) in un_graph.edges(data='weight', default=1):
        if is_undirected:
            continue

        if np.random.uniform() <= train_ratio:
            test_graph.remove_edge(st, ed)
        else:
            train_graph.remove_edge(st, ed)
    return train_graph, test_graph

def node2vec_embedding(graph, name, s):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes (), n=num_walks, length=walk_length, p=p, q=q)
    print (f"Number of random walks for '{name}': {len (walks)}")

    model = Word2Vec(
        walks,
        size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        iter=num_iter,
    )
    idx = list(map(str, model.wv.index2word))
    new_s = list(s[x] for x in idx)
    vec_emb = model.wv.vectors

    def get_embedding(u):
        return model.wv[u]

    return get_embedding, vec_emb, new_s