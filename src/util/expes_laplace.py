import networkx as nx
from util.main_program import *
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import time

# 150 nodes - 2 communities balanced with random protected attributes
method = 'partition'
probs = [[0.20, 0.005], [0.005, 0.10]]


for size in [300]:
    sizes = [size, size]

    print('Size: '+str(size))
    g, s = get_graph_prot(sizes=sizes, probs=probs, choice=method, shuffle=0)

    # t0 = time.time()
    # new_x_l, s, gamma, M = total_repair_reg(g, metric='sqeuclidean', method="laplace", reg=10, case='bin', log=False,
    #                                        name='plot_cost_gamma')
    #
    # print('Elapsed time Laplace: '+str(time.time()-t0))

    t0 = time.time()
    new_x_l_s, s_s, gamma_s, M_s = total_repair_reg(g, metric='sqeuclidean', method="laplace_sinkhorn", reg=1,
                                                    eta = 10, case='bin', log=False, name='plot_cost_gamma')

    print('Elapsed time Laplace Sinkhorn: '+str(time.time()-t0))

    t0 = time.time()
    new_x_l_s, s_s, gamma_s, M_s = total_repair_reg(g, metric='sqeuclidean', method="laplace_traj", reg=10,
                                                    eta = 10, case='bin', log=False, name='plot_cost_gamma')

    print('Elapsed time Laplace Traj: '+str(time.time()-t0))


"""
trial = 1

auc_original = []
auc_rep = []
auc_rep_emb = []

# Laplacian

new_x, gamma, M = total_repair_emd(g, metric='euclidean', case='weighted', log=False, name='plot_cost_gamma')
new_x_l, s, gamma, M = total_repair_reg(g, metric='euclidean', method="laplace", reg=10, case='bin', log=False,
                                       name='plot_cost_gamma')
new_x_traj, s_traj, gamma_traj, M_traj = total_repair_reg(g, metric='euclidean', method="laplace_traj",
                                                     reg=10, case='bin', log=False, name='plot_cost_gamma')


new_g = nx.from_numpy_matrix(new_x)
new_g_l = nx.from_numpy_matrix(new_x_l)
new_g_l_traj = nx.from_numpy_matrix(new_x_traj)

titles = ['Original', 'OT repair', 'Laplacian repair', 'Ferradans repair']

graphs = [g, new_g, new_g_l, new_g_l_traj]

# Plot the networks
prot0 = np.where(s == 0)[0]
prot1 = np.where(s == 1)[0]

plt.figure()
pos = nx.spring_layout(g)
nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot0, node_color='steelblue', label='S = 0')
nx.draw_networkx_nodes(g, pos=pos, node_size=80, nodelist=prot1, node_color='gold', label='S = 1')
nx.draw_networkx_edges(g, pos=pos)
plt.tight_layout()
plt.legend(scatterpoints=1, prop={'size': 15})

plt.figure()
pos = nx.spring_layout(new_g)
nx.draw_networkx_nodes(new_g, pos=pos, node_size=80, nodelist=prot0, node_color='steelblue', label='S = 0')
nx.draw_networkx_nodes(new_g, pos=pos, node_size=80, nodelist=prot1, node_color='gold', label='S = 1')
nx.draw_networkx_edges(new_g, pos=pos)
plt.tight_layout()
plt.legend(scatterpoints=1, prop={'size': 15})


embedsAll = [emb_node2vec(graph, s) for graph in graphs]
embeds = [embedsAll[i][0] for i in range(len(embedsAll))]
ss = [embedsAll[i][1] for i in range(len(embedsAll))]

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)

tsnes = [tsne.fit_transform(embedding) for embedding in embeds]

c = ['steelblue', 'gold']
k = 0
legend_elements = [Line2D([0], [0], marker='o', color='white', label='S = 0', markerfacecolor='steelblue', markersize=15),
                   Line2D([0], [0], marker='o', color='white', label='S = 1',  markerfacecolor='gold', markersize=15)]

fig, ax = plt.subplots(1, len(tsnes), figsize=(20, 8))

for j,t in enumerate(tsnes):
    for g in np.unique(ss[j]):
        i = np.where(ss[j] == g)
        ax[j].scatter(t[:, 0][i], t[:, 1][i], color=c[k], s=100)
        k += 1
    k = 0
    ax[j].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                     labelleft='off')
    ax[j].set_title(titles[j])


#plt.tight_layout()
#plt.savefig('embbedding_g3' + '.eps', bbox_inches='tight')

plt.show()
"""