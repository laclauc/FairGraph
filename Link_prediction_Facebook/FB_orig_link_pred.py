import facebook
import pickle as pkl
import networkx
from src.util.link_prediction import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score

# load the network
facebook.load_network()
# print(facebook.network.order())
# print(facebook.network.size())
adj = networkx.adjacency_matrix(facebook.network)

# Look at a node's features
j = 0
gender = []
protSnum = np.empty(4039)
protSnum2 = np.empty(4039)
list_node = np.asarray(facebook.network.nodes) # order of nodes
list_node_t = map(str, list_node)
protS = dict.fromkeys(list_node_t)

for i in list_node:
    _temp = facebook.network.nodes[i]['features']
#    g = np.asarray[_temp[77], _temp[78]]
    if (_temp[77] == 1 and _temp[78] == 0) or _temp[77] == 2:
        protS[str(i)] = 1
        protSnum[i] = 1
        protSnum2[j] = 1
    else:
        protS[str(i)] = 0
        protSnum[i] = 0
        protSnum2[j] = 0
    j += 1

# model n2v
model = Word2Vec.load('FB_orig_n2v.model')
idx = list(map(str, model.wv.index2word))
X = model.wv.vectors
new_S = [protS[i] for i in idx] 

# train, test datasets
train_edges  = pkl.load(open("train_edges_FB.p", "rb" ) )
test_edges   = pkl.load(open("test_edges_FB.p", "rb" ) )

train_links  = pkl.load(open("train_links_FB.p", "rb" ) )
test_links   = pkl.load(open("test_links_FB.p", "rb" ) )

# hadamard product
train_data = hadamard_fb(model,train_edges,train_links,protS)
test_data  = hadamard_fb(model,test_edges,test_links,protS)

# data preprocessing for link prediction
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
# f.write(str(classification_report(testy, y_pred)))
# f.write('\n\n')
cm = confusion_matrix(testy,y_pred)
print('Confusion Matrix')
print(cm)
print('\nAUC: '+str(auc))

#Disparate Impact
y_true = testy # true labels
#y_pred is predicted labels

#abs_diff_test
same_gender_count = 0
opp_gender_count = 0

index=[]
c=0
for i in y_pred:
    if i==1:
        index.append(c)
    c+=1

for ind in index:
    if abs_diff_test[ind]==0:
        same_gender_count+=1
    else:
        opp_gender_count+=1


print('same_gender_count: ', same_gender_count)
print('opp_gender_count: ', opp_gender_count)

print('Disparate Impact: ', opp_gender_count/same_gender_count)