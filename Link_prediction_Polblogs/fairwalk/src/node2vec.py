import numpy as np
import networkx as nx
import random


class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]	
			neighbors = sorted(G.neighbors(cur))


			# categorizing gender nodes (class0, class1)
			class0_nodes = [nbrs for nbrs in neighbors if G.nodes[nbrs]['gender'] == 0]
			class1_nodes = [nbrs for nbrs in neighbors if G.nodes[nbrs]['gender'] == 1]
			classified_nbrs = [class0_nodes,class1_nodes]

			# Randomly choosing the neighbor class
			if len(class0_nodes) > 0 and len(class1_nodes) > 0:
				cur_nbrs = sorted(classified_nbrs[random.randint(0,1)])
			else:
				ind =  [i for i in range(len(classified_nbrs)) if len(classified_nbrs[i])!= 0]
				cur_nbrs = classified_nbrs[ind[0]]

			#cur_nbrs = sorted(G.neighbors(cur)) #(from orig code)
			#choosing the next node to walk using alias sampling {cur_nbrs is the selected class(0/1) nbrs} 
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					draw = alias_draw(alias_nodes[cur][0], alias_nodes[cur][1],len(cur_nbrs))
					walk.append(cur_nbrs[draw])
					#walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					#print("total neighbors: ",len(neighbors))
					#print(cur_nbrs)
					prev = walk[-2]
					#print(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])
					draw = alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1],len(cur_nbrs))
					#print(draw)
					next = cur_nbrs[draw]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		neighbors = sorted(G.neighbors(dst))
		# categorizing gender nodes (class0, class1) for considering length for probabilities
		class0_nodes = [nbrs for nbrs in neighbors if G.nodes[nbrs]['gender'] == 0]
		class1_nodes = [nbrs for nbrs in neighbors if G.nodes[nbrs]['gender'] == 1]

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			#elif G.has_edge(dst_nbr, src):
				#unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				if G.nodes[dst_nbr]['gender'] == 0:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/len(class0_nodes))
				else:
					unnormalized_probs.append(G[dst][dst_nbr]['weight']/len(class1_nodes))


				#unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)

		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))] # list: weights for each nbr.
			norm_const = sum(unnormalized_probs) 
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs] # normalizing the above list.
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q, size_nbr):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	#K = len(J)
	K = size_nbr

	kk = int(np.floor(np.random.rand()*K))
	return kk
	'''
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]'''