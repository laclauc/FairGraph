from src.util.link_prediction import *
from src.util.main_program import *
import pickle as pkl
import numpy as np

print("Loading the graph")
_temp = nx.read_gml("data/polblogs.gml")
g = _temp.to_undirected(reciprocal=False, as_view=False)
g.remove_nodes_from(list(nx.isolates(g)))
node_list = list(g.nodes(data='value'))

int_node_list = [(node_list.index(i), i[1]) for i in node_list]
protS_int = convert(int_node_list)

# Attribute as a dictionary
tups = node_list
s = convert(tups)
s_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

nx.set_node_attributes(g, s, 's')

print("Repairing the graph with Laplace")

new_x_l, s, gamma, M = total_repair_reg(g, metric='euclidean', method="laplace", reg=1, case='bin', log=False,
                                     name='plot_cost_gamma')
new_g = nx.from_numpy_matrix(new_x_l)

laplace_graph = [new_g, s]
with open('laplace_graph_1.pkl', 'wb') as outfile:
    pkl.dump(laplace_graph, outfile, pkl.HIGHEST_PROTOCOL)


"""

with open("laplace_graph_05.pkl", "rb") as f:
    mat = pkl.load(f)

new_g = mat[0]

print("Learning embedding")
emb_x, new_s, model = emb_node2vec(new_g, s_arr, filename="laplace_knn_5_reg_1000.model")

idx = list(map(str, model.wv.index2word))
X = model.wv.vectors
new_S = new_s

train_edges = pkl.load(open("data/train_edges_polblogs.p", "rb"))
test_edges = pkl.load(open("data/test_edges_polblogs.p", "rb"))

train_links = pkl.load(open("data/train_links_polblogs.p", "rb"))
test_links = pkl.load(open("data/test_links_polblogs.p", "rb"))

# transform from node name to node id
train_edges_tr = transform_str_to_int(node_list, train_edges)
test_edges_tr = transform_str_to_int(node_list, test_edges)

train_data = hadamard(model, train_edges_tr, train_links, protS_int)
test_data = hadamard(model, test_edges_tr, test_links, protS_int)

# data preparation for link-prediction using Logistic Regression
train_tup, trainy = get_tups_data(train_data)
test_tup, testy = get_tups_data(test_data)

trainX, abs_diff_train = map(list, zip(*train_tup))
testX, abs_diff_test = map(list, zip(*test_tup))


model_LR = LogisticRegression(solver='lbfgs', max_iter=500)
model_LR.fit(trainX, trainy)
y_pred = model_LR.predict(testX)

# predict probabilities
probs = model_LR.predict_proba(testX)

# roc_auc_score
auc = roc_auc_score(testy, probs[:, 1])
print('AUC: ', auc)
print('Classification report:')
print(classification_report(testy, y_pred, digits=5))

cm = confusion_matrix(testy, y_pred)
print('Confusion Matrix')
print(cm)

# Disparate Impact
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
"""