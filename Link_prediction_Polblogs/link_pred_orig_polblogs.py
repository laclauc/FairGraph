from OTAdjacency import *
from src.util.link_prediction import *
import networkx as nx
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors


# Load the graph and the protected attribute
_temp = nx.read_gml("polblogs/polblogs.gml")
g = _temp.to_undirected(reciprocal=False, as_view=False)
g.remove_nodes_from(list(nx.isolates(g)))
node_list = list(g.nodes(data='value'))
node_list_dict = g.nodes(data="value")


with open("laplace_graph_05.pkl", "rb") as f:
    mat = pkl.load(f)

new_g = mat[0]
nx.set_node_attributes(new_g, node_list_dict, 'value')



# Attribute as a dictionary
tups = node_list
dictionary = {}
s = convert(tups)
s_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

#print("Loading models and graph")
model = Word2Vec.load("data/polblog_n2v.model")

# loading graph and saving protected atrribute info for each node in protS
d = nx.read_gml("polblogs/polblogs.gml")
#print(nx.info(d))
# From directed to undirected
g = d.to_undirected(reciprocal=False, as_view=False)
#print(nx.info(g))
# Remove isolated nodes
g.remove_nodes_from(list(nx.isolates(g)))
#print(nx.info(g))
node_list = list(g.nodes(data='value'))
#print(node_list)
# Convert the attribute to a dictionnary
# Driver Code
tups = node_list
dictionary = {}
protS = convert(tups)
prot_arr = np.array([x[1] for x in tups])
adj_g = nx.adjacency_matrix(g)

#emb_verse = pkl.load(open('embeddings_dim32_alpha085_PPR.emb', 'rb'))

print("Loading train and test data")
# Polblogs train-test dataset 
# train_edges, test_edges: are the pairs of nodes (like we have in edgelist)
train_edges = pkl.load(open("data/train_edges_polblogs.p", "rb"))
test_edges = pkl.load(open("data/test_edges_polblogs.p", "rb"))

# train_links, test_links: are the labels for the above pair of nodes (1 if there exists an edge between them, 0 otherwise)
train_links = pkl.load(open("data/train_links_polblogs.p", "rb"))
test_links = pkl.load(open("data/test_links_polblogs.p", "rb"))



"""
# train_data, test_data: hadamard returns tuple of (hadamard product, absolute_diff. between sensitive attribute,link_info: labels for edge present or not)
train_data = hadamard(model, train_edges, train_links, protS)
test_data = hadamard(model, test_edges, test_links,protS)

# data preparation for link-prediction using Logistic Regression
train_tup, trainy = get_tups_data(train_data)
test_tup, testy = get_tups_data(test_data)

trainX, abs_diff_train = map(list, zip(*train_tup))
testX, abs_diff_test = map(list, zip(*test_tup))

print("fitting logistic regression")

# fit a model
model_LR = LogisticRegression(solver='lbfgs')
model_LR.fit(trainX, trainy)
y_pred = model_LR.predict(testX)
# predict probabilities
probs = model_LR.predict_proba(testX)
# roc_auc_score
auc = roc_auc_score(testy, probs[:, 1])
print('AUC: ',auc)
print('Classification report: ')
print(classification_report(testy, y_pred, digits=5))
# f.write(str(classification_report(testy, y_pred)))
# f.write('\n\n')
cm = confusion_matrix(testy, y_pred)
print('Confusion Matrix')
print(cm)

# Disparate Impact
y_true = testy # true labels
#y_pred is predicted labels

#abs_diff_test
same_party_count = 0
opp_party_count = 0

index=[]
c=0
for i in y_pred:
    if i==1:
        index.append(c)
    c+=1

for ind in index:
    if abs_diff_test[ind]==0:
        same_party_count+=1
    else:
        opp_party_count+=1


print('same_party_count: ', same_party_count)
print('opp_party_count: ', opp_party_count)

print('Disparate Impact: ', opp_party_count/same_party_count)



# First run knn on the data
fullX = np.concatenate([trainX, testX])
fullY = np.concatenate([trainy, testy])

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

# consistency value is 0.88 on the original graph
"""