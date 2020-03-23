import facebook
import matplotlib.pyplot as plt
import networkx
import numpy as np
import pickle as pkl
from OTAdjacency import *

# load the network
facebook.load_network()

print(facebook.network.order())
print(facebook.network.size())

# Look at a node's features
j = 0
gender = []
protSnum = np.empty(4039)
protSnum2 = np.empty(4039)
list_node = np.asarray(facebook.network.nodes)
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


print(networkx.density(facebook.network))

adj = networkx.adjacency_matrix(facebook.network)
print(adj.shape)
new_adj, gamma, M = total_repair(adj, protSnum, metric='euclidean', case='weighted', algo='emd', reg=2e-3, log=False)


print(new_adj.shape)
"""
fb = [adj.todense(), protSnum2.astype(int)]
fbrepaired = [new_adj, protSnum2.astype(int)]

with open('fbInput.pkl', 'wb') as outfile:
    pkl.dump(fb, outfile, pkl.HIGHEST_PROTOCOL)

with open('fbInputRep.pkl', 'wb') as outfile:
    pkl.dump(fbrepaired, outfile, pkl.HIGHEST_PROTOCOL)

#print(adj.shape)
"""

# look at the features of the whole network
#print(facebook.feature_matrix())

#plt.figure(figsize=(8, 6))
#networkx.draw(facebook.network, with_labels=False, node_size=20, edge_color='darkslategray')
#plt.legend()
#plt.savefig('graph_facebook.png')
#plt.show()


#plt.figure(figsize=(8, 6))
#plt.imshow(facebook.feature_matrix())
#plt.colorbar()
#plt.show()