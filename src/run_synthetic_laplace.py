from util.main_program import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import sys


# synthetic_case = sys.argv[1]
# trial = int(sys.argv[2])
# log = sys.argv[3]
# algo = sys.argv[4]

def density(A):
    # calculate density
    density = np.count_nonzero(A) / float(A.size)
    return density

def equalize_density(A_new, A_old, prop_add):
    threshold = np.linspace(np.amin(A_new), np.amax(A_new), 100)

    den_old = density(A_old)
    den_new = density(A_new)
    k = 0

    while den_new > (1+prop_add) * den_old:
        tmp = np.copy(A_new)
        tmp[tmp < threshold[k]] = 0
        den_new = density(tmp)
        k += 1

    return tmp

synthetic_case = 'g3'
trial = 1
log = 'True'
algo = 'emd'

reg = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 5, 10, 50]
# reg = [0.001, 0.05, 0.5,  0.3, 0.9, 1, 5, 10, 30, 50]

if synthetic_case == 'g1':
    method = 'partition'
    sizes = [75, 75]
    probs = [[0.10, 0.005], [0.005, 0.10]]
    number_class = 'binary'

elif synthetic_case == 'g2':
    method = 'random'
    sizes = [75, 75]
    probs = [[0.10, 0.005], [0.005, 0.10]]
    number_class = 'binary'

elif synthetic_case == 'g3':
    method = 'partition'
    sizes = [125, 25]
    probs = [[0.15, 0.005], [0.005, 0.35]]
    number_class = 'binary'

elif synthetic_case == 'g4':
    method = 'partition'
    probs = [[0.20, 0.002, 0.003], [0.002, 0.15, 0.003], [0.003, 0.003, 0.10]]
    sizes = [50, 50, 50]
    number_class = 'binary'

elif synthetic_case == 'g5':
    method = 'partition'
    probs = [[0.10, 0.001, 0.002], [0.001, 0.10, 0.002], [0.002, 0.002, 0.10]]
    sizes = [50, 50, 50]
    number_class = "multi"
    print(probs)

auc_origin, auc_repair = [], []
density_old, density_rep = [], []
ass_o, ass_rep = [], []

for i in range(trial):
    print("Starting trial " + str(i))
    # Generate the graph
    g, s = get_graph_prot(sizes=sizes, probs=probs, number_class=number_class,
     choice=method)
    ass_o.append(nx.attribute_assortativity_coefficient(g, 's'))

    s = np.fromiter(s.values(), dtype=int)

    auc_reg = []
    ass_reg = []
    for r in reg:
        if number_class == 'binary':

            # Correct the graph with emd
            print("Correcting the graph with Laplace")
            new_x_l, s, gamma, M = total_repair_reg(g, metric='euclidean', method="laplace", reg=r, case='bin',
                                                     log=False)

            new_g = nx.from_numpy_matrix(new_x_l)

            # Filter out the smallest weights to keep a reasonable density
            list_edge = [(u, v) for (u, v, d) in new_g.edges(data=True) if d['weight'] <= 0.5]
            new_g.remove_edges_from(list_edge)

            # Coefficient of assortativity
            dict_s = {i: s[i] for i in range(0, len(s))}
            nx.set_node_attributes(new_g, dict_s, 's')
            ass_reg.append(nx.attribute_assortativity_coefficient(new_g, 's'))

            # Density
            density_old.append(nx.density(g))
            density_rep.append(nx.density(new_g))
        elif number_class == "multi":
            X0 = []

            X = nx.to_numpy_array(g)
            X0.append(X)
            n, d = X.shape

            classes = np.unique(s)

            idxs = []
            Xs = []
            X_repaired = []
            weights = []
            n_i = []
            Cs = []

            for i in classes:
                idxs.append(np.concatenate (np.where(s == i)))
                n_i.append(len(idxs[-1]))
                b_i = np.random.uniform(0., 1., (n_i[-1],))
                b_i = b_i / np.sum(b_i)  # Dirac weights
                Xs.append(X[idxs[-1]])
                Ks = Xs[-1].dot(Xs[-1].T)
                Cs.append(Ks / Ks.max())
                weights.append(b_i)

            X_init = np.random.normal(0., 1., X.shape)  # initial Dirac locations
            b = np.ones((n,)) / n  # weights of the barycenter (it will not be optimized, only the locations are optimized)
            # lambdast = np.ones((3,)) / 3

            X_bary, log = free_support_barycenter_laplace(measures_locations=Xs, measures_weights=weights, reg_type='disp',
                                                          reg_laplace=r, reg_source=0, X_init=X_init, b=b, log=True,
                                                          metric='sqeuclidean', verbose=True, numItermax=20)
            couplings = log['T']
            X0.append(X_bary)

            for i in range(len(classes)):
                X_repaired.append(np.dot(couplings[i].T, X_bary))

            X0.append(np.concatenate(X_repaired))
            new_adj = np.concatenate(X_repaired)

            new_g = nx.from_numpy_matrix(new_adj)

            # Filter out the smallest weights to keep a reasonable density
            list_edge = [(u, v) for (u, v, d) in new_g.edges(data=True) if d['weight'] <= 5e-3]
            new_g.remove_edges_from(list_edge)
            print(nx.density(g))
            print(nx.density(new_g))
            # Coefficient of assortativity
            dict_s = {i: s[i] for i in range(0, len(s))}
            nx.set_node_attributes(new_g, dict_s, 's')
            ass_reg.append(nx.attribute_assortativity_coefficient(new_g, 's'))

        # Learn embedding
        #print("Start learning embedding on the original graph")
        #embedding_origin, s_origin, modelO = emb_node2vec(g, s)

        print("Start learning embedding on the repaired graph")
        embedding_repair, s_repair, modelR = emb_node2vec(new_g, s)

        # Predict the protected attributes
        if number_class == 'binary':
            kfold = model_selection.KFold(n_splits=5, random_state=100, shuffle=True)
            model_kfold = LogisticRegression(solver='lbfgs')

            #results_origin = model_selection.cross_val_score(model_kfold, embedding_origin, s_origin, cv=kfold,
            #                                             scoring='roc_auc')

            #print("AUC on the original graph is: %8.2f" % results_origin.mean())
            #auc_origin.append(results_origin.mean())

            results_repair = model_selection.cross_val_score(model_kfold, embedding_repair, s_repair, cv=kfold,
                                                         scoring='roc_auc')

            print("AUC on the repaired graph is: %8.2f" % results_repair.mean())
            auc_reg.append(results_repair.mean())

        elif number_class == 'multi':
            s_repair = preprocessing.label_binarize(s_repair, classes=[0, 1, 2])

            model = OneVsRestClassifier(LogisticRegression(max_iter=5000))
            params = {'estimator__C': 100. ** np.arange(-1, 1), }
            clf = GridSearchCV(model, params, cv=5, scoring='roc_auc')
            clf.fit(embedding_repair, s_repair)
            auc_reg.append(clf.best_score_)

        print(auc_reg)
        print(ass_reg)

    x = range(10)
    plt.plot(x, auc_reg, marker='o', markersize=8, color='crimson', linestyle='solid', linewidth=2, label='RB')
    plt.plot(x, ass_reg, marker='s', markersize=8, color='steelblue', linestyle='dashdot', linewidth=2, label='Ass.')
    plt.axhline(y=0.95, color='crimson', linestyle='solid', label='RB O.')
    plt.axhline(y=0.74, color='steelblue', linestyle='dashdot', label='Ass. O.')
    plt.xticks(range(10), ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Laplacian regularization', fontsize=15)
    plt.legend(prop={'size': 15}, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    plt.tight_layout()
    plt.savefig('laplace_g5.eps')
    plt.show()

    auc_repair.append(np.asarray(auc_reg).min())
    ass_rep.append(np.asarray(ass_reg).min())


print("Done ! ")
# print("Average AUC over %5d trials on the original graph: %8.2f " % (trial, np.asarray(auc_origin).mean()))
print("Average AUC over %5d trials on the repaired graph: %8.2f " % (trial, np.asarray(auc_repair).mean()))
# print("Average assortativity over %5d trials on the original graph: %8.2f " % (trial, np.asarray(ass_o).mean()))
print("Average assortativity over %5d trials on the repaired graph: %8.2f " % (trial, np.asarray(ass_rep).mean()))

all_results = [auc_repair, ass_rep]
with open(synthetic_case + '_laplace_node2vec.pkl', 'wb') as outfile:
    pkl.dump(all_results, outfile, pkl.HIGHEST_PROTOCOL)
