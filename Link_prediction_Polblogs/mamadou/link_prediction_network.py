import networkx as nx
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
import numpy as np


def convert(tup):
    di = dict(tup)
    return di


def hadamard(model, data, link_info, protS_int):
    hadamard_links = []
    for i in range(len(data)):
        had_pro = np.multiply(model.wv[str(data[i][0])], model.wv[str(data[i][1])])
        absolute_diff = abs(protS_int[data[i][0]]-protS_int[data[i][1]])
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


def compute_di(y_pred):

    # abs_diff_test
    same_party_count = 0
    opp_party_count = 0

    index = []
    c = 0
    for i in y_pred:
        if i == 1:
            index.append(c)
        c += 1

    for ind in index:
        if abs_diff_test[ind] == 0:
            same_party_count += 1
        else:
            opp_party_count += 1

    DI = opp_party_count / same_party_count

    # print('same_party_count: ', same_party_count)
    # print('opp_party_count: ', opp_party_count)

    # print('Disparate Impact: ', opp_party_count / same_party_count)
    return DI


def compute_consistency(test_X, y_pred):
    # First run knn on the data
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(test_X)
    distances, indices = nbrs.kneighbors(test_X)

    n = len(test_X)

    _temp_c = []
    for i in range(n):
        y = y_pred[i]
        idx_neighbors = indices[i]
        list_lab = y_pred[idx_neighbors]
        _temp_c.append(np.sum(np.abs(list_lab - y)))

    cons = 1 - (1 / (n * 11)) * np.sum(_temp_c)
    return cons


# Load the graph and the protected attribute
_temp = nx.read_gml("polblogs/polblogs.gml")
g = _temp.to_undirected(reciprocal=False, as_view=False)
g.remove_nodes_from(list(nx.isolates(g)))
node_list = list(g.nodes(data='value'))

# Attribute as a dictionary
tups = node_list
dictionary = {}
s = convert(tups)
s_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

# Load the node2vec model
model = Word2Vec.load("data/polblog_n2v.model")

tups = node_list
dictionary = {}
protS = convert(tups)
prot_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

print("Loading train and test data")
# train_edges, test_edges: are the pairs of nodes (like we have in edgelist)
train_edges = pkl.load(open("data/train_edges_polblogs.p", "rb"))
test_edges = pkl.load(open("data/test_edges_polblogs.p", "rb"))

# train_links, test_links: are the labels for the above pair of nodes
train_links = pkl.load(open("data/train_links_polblogs.p", "rb" ) )
test_links = pkl.load(open("data/test_links_polblogs.p", "rb" ) )

# train_data, test_data: hadamard returns tuple of (hadamard product, absolute_diff. between sensitive attribute,
# link_info: labels for edge present or not)

train_data = hadamard(model, train_edges, train_links, protS)
test_data = hadamard(model, test_edges, test_links,protS)

# data preparation for link-prediction using Logistic Regression
train_tup, trainy = get_tups_data(train_data)
test_tup, testy = get_tups_data(test_data)

trainX, abs_diff_train = map(list, zip(*train_tup))
testX, abs_diff_test = map(list, zip(*test_tup))

print("Fitting logistic regression")

# fit a model
model_LR = LogisticRegression(solver='lbfgs')
model_LR.fit(trainX, trainy)
y_pred = model_LR.predict(testX)
# predict probabilities
probs = model_LR.predict_proba(testX)
# roc_auc_score
auc = roc_auc_score(testy, probs[:, 1])
print('AUC: ', auc)
print('Classification report: ')
print(classification_report(testy, y_pred, digits=5))

cm = confusion_matrix(testy, y_pred)
print('Confusion Matrix')
print(cm)


disparate_impact = compute_di(y_pred)
print('Disparate Impact: ', disparate_impact)

consistency = compute_consistency(testX, y_pred)
print('Consistency: ', consistency)
