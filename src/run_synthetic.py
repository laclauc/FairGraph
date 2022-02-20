from sklearn.manifold import TSNE
from util.main_program import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from matplotlib.lines import Line2D
import sys


# synthetic_case = sys.argv[1]
# trial = int(sys.argv[2])
# log = sys.argv[3]
# algo = sys.argv[4]


synthetic_case = 'g5'
trial = 20
log = 'False'
algo = 'emd'


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
    probs = [[0.20, 0.002, 0.003], [0.002, 0.15, 0.003], [0.003, 0.003, 0.10]]
    sizes = [50, 50, 50]
    number_class = "multi"

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
    if log == 'True' and i == 1:
        if synthetic_case != 'g5':
            prot0 = np.where(s == 0)[0]
            prot1 = np.where(s == 1)[0]

            pos = nx.spring_layout(g)
            nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot0, node_color='steelblue', label='S = 0')
            nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot1, node_color='gold', label='S = 1')
            nx.draw_networkx_edges(g, pos=pos)
            plt.legend(loc="upper left", scatterpoints=1, prop={'size': 15})
            plt.tight_layout()
            plt.savefig('results/'+synthetic_case+'.eps', bbox_inches='tight', format='eps')
            plt.show()

        elif synthetic_case == 'g5':
            prot0 = np.where(s == 0)[0]
            prot1 = np.where(s == 1)[0]
            prot2 = np.where(s == 2)[0]

            pos = nx.spring_layout(g)
            nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot0, node_color='steelblue', label='S = 0')
            nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot1, node_color='gold', label='S = 1')
            nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot2, node_color='firebrick', label='S = 2')
            nx.draw_networkx_edges(g, pos=pos)
            plt.legend(loc="upper left", scatterpoints=1, prop={'size': 15})
            plt.tight_layout()
            plt.savefig('results/'+synthetic_case+'.eps', bbox_inches='tight', format='eps')
            plt.show()

    if number_class == 'binary':

        # Correct the graph with emd
        print("Correcting the graph with EMD")
        new_adj, s, gamma, M = total_repair_emd(g,  metric='euclidean',
        case='weighted', log=False)
        new_g = nx.from_numpy_matrix(new_adj)

        # Filter out the smallest weights to keep a reasonable density
        list_edge = [(u, v) for (u, v, d) in new_g.edges(data=True) if d['weight'] <= 0.5]
        new_g.remove_edges_from (list_edge)

        # Coefficient of assortativity
        dict_s = {i: s[i] for i in range(0, len(s))}
        nx.set_node_attributes(new_g, dict_s, 's')
        ass_rep.append(nx.attribute_assortativity_coefficient(new_g, 's'))

        # Density
        density_old.append(nx.density(g))
        density_rep.append(nx.density(new_g))
    elif number_class == "multi":
        X0 = []
        X = nx.to_scipy_sparse_matrix(g)
        if issparse(X):
            X = X.todense()
        X = np.squeeze(np.asarray (X))
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
        lambdast = np.ones((3,)) / 3

        X_bary, log = ot.lp.free_support_barycenter(measures_locations=Xs, measures_weights=weights, X_init=X_init,
                                                     b=b, log=True, metric='euclidean')
        couplings = log['T']
        # X0.append(X_bary)

        for k in range(len(classes)):
            X_repaired.append(np.dot(couplings[k].T, X_bary))

        # X0.append(np.concatenate(X_repaired))

        new_adj = np.concatenate(X_repaired)

        new_g = nx.from_numpy_matrix(new_adj)
        # Filter out the smallest weights to keep a reasonable density
        list_edge = [(u, v) for (u, v, d) in new_g.edges(data=True) if d['weight'] <= 5e-3]
        new_g.remove_edges_from(list_edge)
        print (nx.density (g))
        print (nx.density (new_g))
        # Coefficient of assortativity
        dict_s = {i: s[i] for i in range(0, len(s))}
        nx.set_node_attributes(new_g, dict_s, 's')
        ass_rep.append(nx.attribute_assortativity_coefficient(new_g, 's'))

    # Learn embedding
    print("Start learning embedding on the original graph")
    embedding_origin, s_origin, modelO = emb_node2vec(g, s)

    print("Start learning embedding on the repaired graph")
    embedding_repair, s_repair, modelR = emb_node2vec(new_g, s)

    if log == 'True':
        # display TSNE scatter plot
        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)

        Y = tsne.fit_transform(embedding_origin)
        Z = tsne.fit_transform(embedding_repair)
        x1_coords = Y[:, 0]
        y1_coords = Y[:, 1]

        x2_coords = Z[:, 0]
        y2_coords = Z[:, 1]

        c = ['steelblue', 'gold', 'crimson']
        k = 0
        legend_elements = [Line2D([0], [0], marker='o', color='white', label='S = 0', markerfacecolor='steelblue', markersize=15),
                           Line2D([0], [0], marker='o', color='white', label='S = 1',  markerfacecolor='gold', markersize=15),
                           Line2D([0], [0], marker='o', color='white', label='S = 2', markerfacecolor='crimson',
                                   markersize=15)
                           ]

        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        for g in np.unique(s_origin):
            i = np.where(s_origin == g)
            ax[0].scatter(x1_coords[i], y1_coords[i], color=c[k], s=100)
            k += 1
        k = 0
        ax[0].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                         labelleft='off')
        for g in np.unique(s_repair):
            i = np.where(s_repair == g)
            ax[1].scatter(x2_coords[i], y2_coords[i], color=c[k], s=100)
            k += 1

        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                  ncol=3, fancybox=True, shadow=False, prop={'size': 30}, fontsize=30)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                         labelleft='off')
        plt.tight_layout()
        plt.savefig(synthetic_case + '.eps', bbox_inches='tight')
        plt.show()

    # Predict the protected attributes
    if number_class == 'binary':
        kfold = model_selection.KFold(n_splits=5, random_state=100, shuffle=True)
        model_kfold = LogisticRegression(solver='lbfgs')

        results_origin = model_selection.cross_val_score(model_kfold, embedding_origin, s_origin, cv=kfold,
                                                     scoring='roc_auc')

        print("AUC on the original graph is: %8.2f" % results_origin.mean())
        auc_origin.append(results_origin.mean())

        results_repair = model_selection.cross_val_score(model_kfold, embedding_repair, s_repair, cv=kfold,
                                                     scoring='roc_auc')

        print("AUC on the repaired graph is: %8.2f" % results_repair.mean())
        auc_repair.append(results_repair.mean())
    elif number_class == 'multi':
        s_origin = preprocessing.label_binarize(s_origin, classes=[0, 1, 2])
        s_repair = preprocessing.label_binarize(s_repair, classes=[0, 1, 2])

        model = OneVsRestClassifier(LogisticRegression(max_iter=5000))
        params = {'estimator__C': 100. ** np.arange(-1, 1), }
        clf = GridSearchCV(model, params, cv=5, scoring='roc_auc')
        clf.fit(embedding_origin, s_origin)
        auc_origin.append(clf.best_score_)

        model = OneVsRestClassifier(LogisticRegression(max_iter=5000))
        params = {'estimator__C': 100. ** np.arange(-1, 1), }
        clf = GridSearchCV(model, params, cv=5, scoring='roc_auc')
        clf.fit(embedding_repair, s_repair)
        auc_repair.append(clf.best_score_)

print("Done ! ")
print("Average AUC over %5d trials on the original graph: %8.2f " % (trial, np.asarray(auc_origin).mean()))
print("Average AUC over %5d trials on the repaired graph: %8.2f " % (trial, np.asarray(auc_repair).mean()))
print("Std AUC over %5d trials on the original graph: %8.2f " % (trial, np.asarray(auc_origin).std()))
print("Std AUC over %5d trials on the repaired graph: %8.2f " % (trial, np.asarray(auc_repair).std()))
print("Average assortativity over %5d trials on the original graph: %8.2f " % (trial, np.asarray(ass_o).mean()))
print("Average assortativity over %5d trials on the repaired graph: %8.2f " % (trial, np.asarray(ass_rep).mean()))
print("Std assortativity over %5d trials on the original graph: %8.2f " % (trial, np.asarray(ass_o).std()))
print("Std assortativity over %5d trials on the repaired graph: %8.2f " % (trial, np.asarray(ass_rep).std()))

all_results = [auc_origin, auc_repair]
with open(synthetic_case + '_graph_node2vec.pkl', 'wb') as outfile:
    pkl.dump(all_results, outfile, pkl.HIGHEST_PROTOCOL)
