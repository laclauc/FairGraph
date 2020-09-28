from src.util.main_program import *
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read author-author edge list and protected attribute
g = nx.read_edgelist("data/author-author.edgelist")
node_id = list(g.nodes)

# Read attribute
df = pd.read_csv("data/countries.attributes", delimiter=":", header=None)
df.columns = ["nodeId", "Country"]

# Convert nodeId to string to match the graph and encode the country
df["nodeId"] = df["nodeId"].astype(str)

LE = LabelEncoder()
df['Country'] = LE.fit_transform(df['Country'])

protS = dict(zip(df['nodeId'], df['Country']))

nx.set_node_attributes(g, protS, name='protS')


# Remove node in protS which are not in the graph (isolated nodes)
new = protS.copy()
for key, value in protS.items():
    if key not in node_id:
        new.pop(key)

protS = new.copy()
protS_array = np.array(list(protS.values()))

X0 = []

X = nx.to_numpy_array(g)
X0.append(X)
n, d = X.shape

classes = np.unique(protS_array)

idxs = []
Xs = []
X_repaired = []
weights = []
n_i = []
Cs = []

for i in classes:
    idxs.append(np.concatenate(np.where(protS_array == i)))
    n_i.append(len(idxs[-1]))
    b_i = np.random.uniform(0., 1., (n_i[-1],))
    b_i = b_i / np.sum(b_i)  # Dirac weights
    Xs.append(X[idxs[-1]])
    Ks = Xs[-1].dot(Xs[-1].T)
    Cs.append(Ks/Ks.max())
    weights.append(b_i)

X_init = np.random.normal(0., 1., X.shape)  # initial Dirac locations
b = np.ones((n,)) / n  # weights of the barycenter (it will not be optimized, only the locations are optimized)
# lambdast = np.ones((3,)) / 3

X_bary, log = ot.lp.free_support_barycenter(measures_locations=Xs, measures_weights=weights, X_init=X_init, b=b,
                                            log=True, metric='sqeuclidean')
couplings = log['T']
X0.append(X_bary)

for i in range(len(classes)):
    X_repaired.append(np.dot(couplings[i].T, X_bary))

X0.append(np.concatenate(X_repaired))
new_adj = np.concatenate(X_repaired)

new_g = nx.from_numpy_matrix(new_adj)

repaired_graph = [new_g, protS]
with open('dblp_emd_graph.pkl', 'wb') as outfile:
    pkl.dump(repaired_graph, outfile, pkl.HIGHEST_PROTOCOL)
