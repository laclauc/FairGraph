from OTAdjacency import *
import networkx as nx
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec


def Convert(tup):
    di = dict(tup)
    return di


def hadamard(model, data, link_info, protS_int):
    hadamard_links = []
    for i in range(len(data)):
        had_pro = np.multiply(model.wv[str(data[i][0])], model.wv[str(data[i][1])])  # hadamard product
        absolute_diff = abs(protS_int[data[i][0]]-protS_int[data[i][1]])
        hadamard_links.append((had_pro, absolute_diff, link_info[i]))
    return hadamard_links


def get_tups_data(hadamard_data):
    vectors_and_abs_val = []
    links=[]
    for i in hadamard_data:
        vectors_and_abs_val.append((i[0], i[1]))
        links.append(i[2])
    return vectors_and_abs_val, links


def transform_str_to_int(edges):
    for i in range(len(edges)):
        ind_first_ele = [y[0] for y in node_list].index(edges[i][0])
        ind_sec_ele = [y[0] for y in node_list].index(edges[i][1])
        edges[i] = (ind_first_ele, ind_sec_ele)
    return edges


model = Word2Vec.load('polblogs/laplace_knn_3_reg_100.model')

d = nx.read_gml("polblogs/polblogs.gml")
g = d.to_undirected(reciprocal=False, as_view=False)

g.remove_nodes_from(list(nx.isolates(g)))

node_list = list(g.nodes(data='value'))
int_node_list = [(node_list.index(i), i[1]) for i in node_list]
dict_is = {}
protS_int = Convert(int_node_list)

tups = node_list
dictionary = {}
protS = Convert(tups)

prot_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

# indexing protS according to node vectors
idx = list(map(str, model.wv.index2word))
X = model.wv.vectors
new_S = [protS_int[int(i)] for i in idx]


train_edges = pkl.load(open("data/train_edges_polblogs.p", "rb"))
test_edges = pkl.load(open("data/test_edges_polblogs.p", "rb"))

train_links = pkl.load(open("data/train_links_polblogs.p", "rb"))
test_links = pkl.load(open("data/test_links_polblogs.p", "rb"))

train_edges_tr = transform_str_to_int(train_edges)
test_edges_tr = transform_str_to_int(test_edges)

train_data = hadamard(model, train_edges_tr, train_links, protS_int)
test_data = hadamard(model, test_edges_tr, test_links, protS_int)

# data preparation for link-prediction using Logistic Regression
train_tup, trainy = get_tups_data(train_data)
test_tup, testy = get_tups_data(test_data)

trainX, abs_diff_train = map(list, zip(*train_tup))
testX, abs_diff_test = map(list, zip(*test_tup))


# fit a model
model_LR = LogisticRegression(solver='lbfgs')
model_LR.fit(trainX, trainy)
y_pred = model_LR.predict(testX)

probs = model_LR.predict_proba(testX)
# roc_auc_score
auc = roc_auc_score(testy, probs[:, 1])
print('AUC: ', auc)
print('/nClassification report: /n')
print(classification_report(testy, y_pred, digits=5))

cm = confusion_matrix(testy, y_pred)
print('Confusion Matrix')
print(cm)

y_true = testy

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


print('same_party_count: ', same_party_count)
print('opp_party_count: ', opp_party_count)

print('Disparate Impact: ', opp_party_count/same_party_count)

# Compute consistency measure

nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(testX)
distances, indices = nbrs.kneighbors(testX)

n = len(testX)

_temp = []
for i in range(n):
    y = y_pred[i]
    idx_neighbors = indices[i]
    list_lab = y_pred[idx_neighbors]
    _temp.append(np.sum(np.abs(list_lab-y)))

consistency = 1-(1/(n*11))*np.sum(_temp)

print(consistency)

# knn 5 and reg 0.05 : 0.77
# knn 3 and reg 100 : 0.80

