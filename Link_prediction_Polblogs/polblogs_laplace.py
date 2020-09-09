from src.util.link_prediction import *
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
import multiprocessing
from sklearn.model_selection import train_test_split
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle as pkl


def Convert(tup):
    dic = dict(tup)
    return dic

d = nx.read_gml("polblogs/polblogs.gml")
g = d.to_undirected(reciprocal=False, as_view=False)

g.remove_nodes_from(list(nx.isolates(g)))

node_list = list(g.nodes(data='value'))
lab_node_list = [(node_list.index(i), i[0]) for i in node_list]
lab_node_array = np.array(lab_node_list)

int_node_list = [(node_list.index(i), i[1]) for i in node_list]
dict_is = {}
protS_int = Convert(int_node_list)

tups = node_list
dictionary = {}
protS = Convert(tups)

prot_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

with open("laplace_graph_05.pkl", "rb") as f:
    mat = pkl.load(f)

new_graph = mat[0]

list_edge = [(u, v) for (u, v, d) in new_graph.edges(data=True) if d['weight'] < 0.45]
new_graph.remove_edges_from(list_edge)
lab = {k: j for k,j in zip(new_graph.nodes, lab_node_array[:, 1])}
h = nx.relabel_nodes(new_graph, lab)

# Using Stellar library
stellar_polblogs_lap = StellarGraph.from_networkx(h)
stellar_polblogs = StellarGraph.from_networkx(g)

auc, di, cons, rep_bias = [], [], [], []
auc_train = []
trials = 10

for i in range(trials):

    # Define an edge splitter on the corrected graph:
    edge_splitter_test = EdgeSplitter(stellar_polblogs_lap)
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.3, method="global",
                                                                                  keep_connected=True)
    # Do the same process to compute a training subset from within the test graph
    edge_splitter_train = EdgeSplitter(graph_test, stellar_polblogs_lap)
    graph_train, examples, labels = edge_splitter_train.train_test_split(p=0.3, method="global", keep_connected=True)

    (
        examples_train,
        examples_model_selection,
        labels_train,
        labels_model_selection,
    ) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

    for k,i in enumerate(examples_train):
        tup = (i[0],i[1])
        labels_test[k] = int(g.has_edge(*tup))

    """
    # Clear examples and examples_test by removing tuples which do not exist in the original graph
    # First only keep positive labels on both sets
    idx_ex_pos = np.where(labels == 1)
    idx_test_pos = np.where(labels_test == 1)

    examples_pos = examples[idx_ex_pos]
    examples_test_pos = examples_test[idx_test_pos]

    # Second : concatenate both
    full_pos = np.concatenate((examples_pos, examples_test_pos), axis=0)

    # Third : remove "false positive" based on the original graph tuples
    # Identify fake positive
    true_pos = nx.to_pandas_edgelist(g)
    true_pos = true_pos.values

    list_full_pos = list(map(tuple, full_pos))
    list_true_pos = list(map(tuple, true_pos))

    _temp = set(map(frozenset, list_full_pos)) & set(map(frozenset, list_true_pos))
    res_fi = [tuple(element) for element in _temp]

    # Fourth : Remove fake negative from examples and examples_test by comparing full pos with true_pos
    """

    p = 2
    q = 2
    dimensions = 128
    num_walks = 10
    walk_length = 35
    window_size = 10
    num_iter = 10
    workers = multiprocessing.cpu_count()


    def node2vec_embedding(graph, name, s):
        rw = BiasedRandomWalk(graph)
        walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
        print(f"Number of random walks for '{name}': {len(walks)}")

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


    print("Start computing absolute differences between protected attributes for each set")
    # Compute absolute difference for the protected attribute

    def abs_diff(examples, S):
        len_ex = examples.shape[0]
        absolute_diff = []
        for i in range(len_ex):
            absolute_diff.append(abs(S[str(examples[i][0])] - S[str(examples[i][1])]))
        return absolute_diff


    abs_diff_train = abs_diff(examples_train, protS)
    abs_diff_model_selection = abs_diff(examples_model_selection, protS)
    abs_diff_test = abs_diff(examples_test, protS)

    print("Start node2vec on the graph train")
    embedding_train, vec_train, s_train = node2vec_embedding(graph_train, "Train Graph", protS)

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

        same_group_count = 0
        opp_group_count = 0

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
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(link_features)
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


    def run_link_prediction(binary_operator):
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


    def representation_bias(ex_train, ex_test, label_train, label_test):
        lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=1000).fit(ex_train, label_train)
        predicted = lr_clf.predict_proba(ex_test)

        # check which class corresponds to positive links
        positive_column = list(lr_clf.classes_).index(1)
        return roc_auc_score(label_test, predicted[:, positive_column])

    binary_operators = [operator_hadamard]
    results = [run_link_prediction(op) for op in binary_operators]
    best_result = max(results, key=lambda result: result["score"])

    print(f"Best result from '{best_result['binary_operator'].__name__}'")

    print(pd.DataFrame(
        [(result["binary_operator"].__name__, result["score"], result["disparate_impact"]) for result in results],
        columns=("name", "ROC AUC score", "DI score"),).set_index("name"))

    auc_train.append(best_result['score'])

    print("Start node2vec on the graph test")
    embedding_test, vec_test, s_test = node2vec_embedding(graph_test, "Test Graph", protS)

    auc_protS = representation_bias(vec_train, vec_test, s_train, s_test)
    rep_bias.append(auc_protS)
    print(rep_bias)
    test_score, test_score_bias, test_score_consistency = evaluate_link_prediction_model(
        best_result["classifier"],
        examples_test,
        labels_test,
        embedding_test,
        best_result["binary_operator"],
        abs_diff_test
    )
    print(f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}")
    print(f"DI score on test set using '{best_result['binary_operator'].__name__}': {test_score_bias}")
    print(f"Consistency score on test set using '{best_result['binary_operator'].__name__}': {test_score_consistency}")

    auc.append(test_score)
    di.append(test_score_bias)
    cons.append(test_score_consistency)

print("Done ! ")

print("Average AUC over 10 trials: %8.2f (%8.2f) " % (np.asarray(auc).mean(), np.asarray(auc).std()))
print("Average DI over 10 trials: %8.2f (%8.2f) " % (np.asarray(di).mean(), np.asarray(di).std()))
print("Average Consistency over 10 trials: %8.2f (%8.2f) " % (np.asarray(cons).mean(), np.asarray(cons).std()))
print("Average Representation Bias over 10 trials: %8.2f (%8.2f) " % (np.asarray(rep_bias).mean(),
                                                                      np.asarray(rep_bias).std()))

all_results = [auc, di, cons, rep_bias]
with open('results/polblogs_node2vec_laplace05.pkl', 'wb') as outfile:
    pkl.dump(all_results, outfile, pkl.HIGHEST_PROTOCOL)

