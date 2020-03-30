from src.util.main_program import *
from src.util.link_prediction import *
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
import sys


print('Loading the graph')
_temp = nx.read_gml("polblogs/polblogs.gml")
g = _temp.to_undirected(reciprocal=False, as_view=False)
g.remove_nodes_from(list(nx.isolates(g)))
node_list = list(g.nodes(data='value'))

# Attribute as a dictionary
tups = node_list
s = convert(tups)
s_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

# Setting the attribute as a feature of the nodes in the graph
nx.set_node_attributes(g, s, 's')

print('Loading pre-trained model')
mode_file = sys.argv[1]
model = Word2Vec.load('model_file')

# polblogs train-test data
# train_edges, test_edges: are the pairs of nodes (like we have in edgelist)
print('Loading train and test data')
train_edges = pkl.load(open("train_edges_polblogs.p", "rb"))
test_edges = pkl.load(open("test_edges_polblogs.p", "rb"))

# train_links, test_links: are the labels for the above pair of nodes (1 if there exists an edge
# between them, 0 otherwise)

train_links = pkl.load(open("train_links_polblogs.p", "rb"))
test_links = pkl.load(open("test_links_polblogs.p", "rb"))

# transform from node name to node id
train_edges_tr = transform_str_to_int(node_list, train_edges)
test_edges_tr = transform_str_to_int(node_list, test_edges)

# train_data, test_data: Hadamard returns tuple of (hadamard product, absolute_diff.
# between sensitive attribute,link_info: labels for edge present or not)

train_data = hadamard(model, train_edges_tr, train_links, s)
test_data = hadamard(model, test_edges_tr, test_links, s)

# data preparation for link-prediction using Logistic Regression
train_tup, trainy = get_tups_data(train_data)
test_tup, testy = get_tups_data(test_data)

trainX, abs_diff_train = map(list, zip(*train_tup))
testX, abs_diff_test = map(list, zip(*test_tup))


# fit a model
model_LR = LogisticRegression(solver='lbfgs')
model_LR.fit(trainX, trainy)
y_pred = model_LR.predict(testX)
# predict probabilities
probs = model_LR.predict_proba(testX)
# roc_auc_score
auc = roc_auc_score(testy, probs[:, 1])
print('AUC: ', auc)
print('/nClassification report: /n')
print(classification_report(testy, y_pred, digits=5))

cm = confusion_matrix(testy, y_pred)
print('Confusion Matrix')
print(cm)

# Disparate Impact
y_true = testy
# true labels
# y_pred is predicted labels

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


print('same_party_count: ', same_party_count)
print('opp_party_count: ', opp_party_count)

print('Disparate Impact: ', opp_party_count/same_party_count)
