from sklearn.model_selection import train_test_split
from OTAdjacency import *
import networkx as nx
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

def Convert(tup, di):
    di = dict(tup)
    return di

d = nx.read_gml("real/polblogs/polblogs.gml")
print(nx.info(d))
# From directed to undirected
g = d.to_undirected(reciprocal=False, as_view=False)
print(nx.info(g))
# Remove isolated nodes
g.remove_nodes_from(list(nx.isolates(g)))
print(nx.info(g))
# getting node list with sensitive attribute values
node_list = list(g.nodes(data='value'))
new_node_list = [(node_list.index(i),i[1]) for i in node_list]
#print(node_list)
# Convert the attribute to a dictionnary
# Driver Code
tups = new_node_list # tuples (node,attr) list
dictionary = {}
protS = Convert(tups, dictionary)
print(protS)
#prot_arr= np.array([x[1] for x in tups])
#print(prot_arr)
#adj_g = nx.adjacency_matrix(g)

    
#for fairwalk write edgelist and write protS in dict form
edge_file = 'polblog_fairwalk_.edgelist'
nx.write_edgelist(g,edge_file,data=False)
dict_file = 'polblogs_dict_attr.npy'
np.save(dict_file, protS)
out_file = 'polblogs_fairwalk_.emb'
#print(protS)
    
# Fairwalk Node embedding on original graph
embeddings_g = fairwalk(input_edgelist=edge_file,output_emb_file=out_file,dict_file=dict_file) #
print('YOOO done!!!')
#visuTSNE(X_old, protS, k=2, seed=0, plotName='tsne_g1_old')
#visuTSNE(X_rep, protS, k=2, seed=42, plotName='tsne_g1_new')




