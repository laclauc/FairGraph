import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.data import BiasedRandomWalk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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
    p = 2
    q = 2
    dimensions = 64
    num_walks = 10
    walk_length = 35
    window_size = 10
    num_iter = 10
    
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes (), n=num_walks, length=walk_length, p=p, q=q)
    print (f"Number of random walks for '{name}': {len (walks)}")

    model = Word2Vec(
        walks,
        size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        iter=num_iter,
    )
    idx = list(map(str, model.wv.index2word))
    new_s = list(s[x] for x in idx)
    vec_emb = model.wv.vectors

    def get_embedding(u):
        return model.wv[u]

    return get_embedding, vec_emb, new_s

def abs_diff(examples, S):
        len_ex = examples.shape[0]
        absolute_diff = []
        for i in range(len_ex):
            absolute_diff.append(abs(S[str(examples[i][0])] - S[str(examples[i][1])]))
        return absolute_diff

def loadPolblog(path, verbose=True):
    _temp = nx.read_gml(path+"polblogs/polblogs.gml")
    g = _temp.to_undirected(reciprocal=False, as_view=False)
    g.remove_nodes_from(list(nx.isolates(g)))
    node_list = list(g.nodes(data='value'))

    lab_node_list = [(node_list.index(i), i[0]) for i in node_list]
    lab_node_array = np.array(lab_node_list)
    
    # Attribute as a dictionary
    tups = node_list
    s = convert(tups)
    s_arr = np.array([x[1] for x in tups])
    adj_g = nx.adjacency_matrix(g)

    # Add attribute to the graph
    nx.set_node_attributes(g, s, 's')
 
    if verbose:
        print(f'Number of nodes: {g.number_of_nodes()}')
        print(f'Number of edges: {g.number_of_edges()}')
        print(f'Average node degree: {g.number_of_edges() / g.number_of_nodes():.2f}')
        print(f'Has isolated nodes: {len(list(nx.isolates(g)))}')
        print(f'Is directed: {nx.is_directed(g)}')

       # Assortativity coefficient 
        print('Assortativity coefficient: %0.3f'
          % nx.attribute_assortativity_coefficient(g, 's'))
    return g, lab_node_array

def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]


def train_link_prediction_model(link_examples, link_labels, get_embedding, binary_operator):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


def evaluate_link_prediction_model(clf, link_examples_test, link_labels_test, get_embedding,
                                   binary_operator, abs_diff_test):

    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    score_bias = evaluate_bias(clf, link_features_test, abs_diff_test)
    score_consistency = evaluate_consistency(clf, link_features_test)
    return score, score_bias, score_consistency


def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

     # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


def evaluate_bias(clf, link_features, abs_diff):
    pred = clf.predict(link_features)

    same_group_count = 1
    opp_group_count = 1

    index = []
    c = 0
    for i in pred:
        if i == 1:
            index.append(c)
        c += 1

    for ind in index:
        if abs_diff[ind] == 0:
            same_group_count += 1
        else:
            opp_group_count += 1

    return opp_group_count / same_group_count


def evaluate_consistency(clf, link_features):
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(link_features)
    distances, indices = nbrs.kneighbors(link_features)
    y_pred = clf.predict(link_features)

    n = len(link_features)

    _temp = []
    for i in range(n):
        y = y_pred[i]
        idx_neighbors = indices[i]
        list_lab = y_pred[idx_neighbors]
        _temp.append(np.sum(np.abs(list_lab - y)))

    consistency = 1 - (1 / (n * 11)) * np.sum(_temp)

    return consistency


def run_link_prediction(binary_operator, examples_train, labels_train, embedding_train,
                       examples_model_selection, labels_model_selection, 
                        abs_diff_model_selection):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score, score_bias, score_consistency = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
        abs_diff_model_selection
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
        "disparate_impact": score_bias,
        "consistency": score_consistency
    }


def representation_bias(ex_train, label_train):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=1000).fit(ex_train, label_train)
    # predicted = lr_clf.predict_proba(ex_test)

    # check which class corresponds to positive links
    # positive_column = list(lr_clf.classes_).index(1)
    # return roc_auc_score(label_train, predicted[:, positive_column])
    return lr_clf.score(ex_train, label_train)

