from OTAdjacency import *
import networkx as nx
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score, roc_curve, f1_score,auc
import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec

def Convert(tup, di):
    di = dict(tup)
    return di

def hadamard(model,data,link_info,protS_int):
    hadamard_links = [] #absolute_diff = [] #absolute diff between sensitive attributes
    for i in range(len(data)):
        had_pro = np.multiply(model.wv[str(data[i][0])],model.wv[str(data[i][1])])  # hadamard product
        absolute_diff = abs(protS_int[data[i][0]]-protS_int[data[i][1]])
        hadamard_links.append((had_pro,absolute_diff,link_info[i]))
    return hadamard_links

def get_tups_data(hadamard_data):
    vectors_and_abs_val=[]
    links=[]
    for i in hadamard_data:
        vectors_and_abs_val.append((i[0],i[1]))
        links.append(i[2])
    return vectors_and_abs_val,links

def transform_str_to_int(orig_node_list, edges):
    '''node_list is the tuple list in order of graph'''
    lis_tup=[]
    for i in range(len(edges)):
        ind_first_ele = [y[0] for y in node_list].index(edges[i][0])
        ind_sec_ele   = [y[0] for y in node_list].index(edges[i][1])
        edges[i] = (ind_first_ele,ind_sec_ele)
    return edges

# Pretrained node2vec on repaired polblogs data
model = Word2Vec.load('repaired_n2v_polblog.model')

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
int_node_list = [(node_list.index(i),i[1]) for i in node_list]
dict_is={}
protS_int = Convert(int_node_list,dict_is)
#print(node_list)
# Convert the attribute to a dictionnary
# Driver Code
tups = node_list
dictionary = {}
protS = Convert(tups, dictionary)
#print(protS)
prot_arr= np.array([x[1] for x in tups])
#print(prot_arr)
adj_g = nx.adjacency_matrix(g)

# indexing protS according to node vectors
idx = list(map(str, model.wv.index2word))
X = model.wv.vectors
new_S = [protS_int[int(i)] for i in idx] 

# Polblogs train-test dataset 
# train_edges, test_edges: are the pairs of nodes (like we have in edgelist)
train_edges  = pkl.load(open("train_edges_polblogs.p", "rb" ) )
test_edges   = pkl.load(open("test_edges_polblogs.p", "rb" ) )

# train_links, test_links: are the labels for the above pair of nodes (1 if there exists an edge between them, 0 otherwise)
train_links  = pkl.load(open("train_links_polblogs.p", "rb" ) )
test_links   = pkl.load(open("test_links_polblogs.p", "rb" ) )

# transform from node name to node id
train_edges_tr = transform_str_to_int(node_list,train_edges)
test_edges_tr = transform_str_to_int(node_list,test_edges)

# train_data, test_data: hadamard returns tuple of (hadamard product, absolute_diff. between sensitive attribute,link_info: labels for edge present or not)
train_data = hadamard(model,train_edges_tr,train_links,protS_int)
test_data  = hadamard(model,test_edges_tr,test_links,protS_int)

# data preparation for link-prediction using Logistic Regression
train_tup,trainy = get_tups_data(train_data)
test_tup,testy = get_tups_data(test_data)

trainX, abs_diff_train = map(list,zip(*train_tup))
testX, abs_diff_test = map(list,zip(*test_tup))


# fit a model
model_LR = LogisticRegression(solver='lbfgs')
model_LR.fit(trainX, trainy)
y_pred = model_LR.predict(testX)
# predict probabilities
probs = model_LR.predict_proba(testX)
# roc_auc_score
auc = roc_auc_score(testy, probs[:, 1])
print('AUC: ',auc)
print('/nClassification report: /n')
print(classification_report(testy, y_pred,digits=5))

cm = confusion_matrix(testy,y_pred)
print('Confusion Matrix')
print(cm)

#Disparate Impact
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

