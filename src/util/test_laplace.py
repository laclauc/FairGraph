import networkx as nx
from util.main_program import *
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
# 150 nodes - 2 communities balanced with random protected attributes
method = 'partition'
sizes = [75, 75]
probs = [[0.20, 0.005], [0.005, 0.10]]

trial = 1

auc_original = []
auc_rep = []
auc_rep_emb = []


g, s = get_graph_prot(sizes=sizes, probs=probs, choice=method, shuffle=0)

# Laplacian

#new_x, gamma, M = total_repair_emd(g, metric='euclidean', case='weighted', log=False, name='plot_cost_gamma')

new_x, s, gamma, M = total_repair_reg(g, metric='euclidean', method="laplace", reg=1, case='bin', log=False,
                                      name='plot_cost_gamma')
new_g = nx.from_numpy_matrix(new_x)
# Trajectory laplacian

#new_x_traj, s_traj, gamma_traj, M_traj = total_repair_reg(g, metric='euclidean', method="laplace_traj",
#                                                     reg=0.01, case='bin', log=False, name='plot_cost_gamma')
#new_g_traj = nx.from_numpy_matrix(new_x_traj)
# Plot the networks

"""
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
"""

embedding_origin, s_origin = emb_node2vec(g, s)
embedding_repair, s_repair = emb_node2vec(new_g, s)

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)

Y = tsne.fit_transform(embedding_origin)
Z = tsne.fit_transform(embedding_repair)
x1_coords = Y[:, 0]
y1_coords = Y[:, 1]

x2_coords = Z[:, 0]
y2_coords = Z[:, 1]

c = ['steelblue', 'gold']
k = 0
legend_elements = [Line2D([0], [0], marker='o', color='white', label='S = 0', markerfacecolor='steelblue', markersize=15),
                   Line2D([0], [0], marker='o', color='white', label='S = 1',  markerfacecolor='gold', markersize=15)]

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
          ncol=2, fancybox=True, shadow=False, prop={'size': 30}, fontsize=30)
#plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
#                 labelleft='off')
#plt.tight_layout()
#plt.savefig('embbedding_g3' + '.eps', bbox_inches='tight')
plt.show()
