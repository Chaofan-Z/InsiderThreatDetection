import itertools
import math
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm import trange

from .alias import alias_sample, create_alias_table
from .utils import partition_num
import pdb

def get_neighbors_edgeType(G, cur, edge_type):
    res = []
    for nbr in G.neighbors(cur):
        # print(G[cur][nbr])
        if G[cur][nbr][0].get('EdgeType', 1) in edge_type:
            res.append(nbr)
    return res


class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=1):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.G = G
        self.p = p
        self.q = q
        self.nodes = []
        for node in G.nodes():
            if G.nodes[node]['A'] == 'Logon':
                self.nodes.append(node)
        print("=" * 20)
        print("Graph start nodes : ", len(self.nodes))
        self.use_rejection_sampling = use_rejection_sampling
    def deepwalk_walk(self, walk_length, start_node, edge_type = [1]):

        walk = [start_node]
        # pdb.set_trace()
        while len(walk) < walk_length:
            cur = walk[-1]
            # cur_nbrs = list(self.G.neighbors(cur))
            cur_nbrs = get_neighbors_edgeType(self.G, cur, edge_type)
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node, edge_type):
        # 
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        has_higher_edge = False
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = get_neighbors_edgeType(self.G, cur, edge_type)
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    if has_higher_edge or G[prev][cur][0]['EdgeType'] == edge_type[-1]:
                        has_higher_edge = True

                    walk.append(next_node)
            else:
                break

        return walk, has_higher_edge

    def node2vec_walk2(self, walk_length, start_node, edge_type = [1]):
        """
        Reference:
        KnightKing: A Fast Distributed Graph Random Walk Engine
        http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf
        """

        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if (inv_p > second_upper_bound):
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        has_higher_edge = False
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            # cur_nbrs = list(G.neighbors(cur))
            cur_nbrs = get_neighbors_edgeType(self.G, cur, edge_type)
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    # if cur == "{L5K1-L1NL17VJ-9414WZTP}":
                    #     print(cur)
                    #     print(alias_nodes[cur])
                    #     print(cur_nbrs)
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(
                        inv_p, inv_q, len(cur_nbrs))
                    prev = walk[-2]
                    # prev_nbrs = set(G.neighbors(prev))
                    prev_nbrs = get_neighbors_edgeType(self.G, prev, edge_type)
                    while True:
                        prob = random.random() * upper_bound
                        if (prob + shatter >= upper_bound):
                            next_node = prev
                            break
                        next_node = cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])]
                        if (prob < lower_bound):
                            break
                        if (prob < inv_p and next_node == prev):
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if (prob < _prob):
                            break

                    if has_higher_edge or G[cur][next_node][0]['EdgeType'] == edge_type[-1]:
                        has_higher_edge = True
                    walk.append(next_node)
            else:
                break

        return walk, has_higher_edge

    def simulate_walks(self, edge_type, num_walks, walk_length, sentence_min_len, workers=1, verbose=0):

        # G = self.G
        # nodes = []
        # for node in G.nodes():
        #     if G.nodes[node]['A'] == 'Logon':
        #         nodes.append(node)
        # nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(self.nodes, num, walk_length, edge_type, sentence_min_len) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, edge_type, sentence_min_len):
        walks = []
        for _ in range(num_walks):
            # random.shuffle(nodes)
            for v in tqdm(nodes):
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v, edge_type=edge_type))
                # reject = 1后 进入这里
                elif self.use_rejection_sampling:
                    # print("-----------node2vec2222222-----------")
                    walk, has_higher_edge = self.node2vec_walk2(walk_length=walk_length, start_node=v, edge_type=edge_type)

                    if len(walk) < sentence_min_len or not has_higher_edge:
                        continue 

                    walks.append(walk)
                else:
                    # print("-----------node2vec-----------")
                    walk, has_higher_edge = self.node2vec_walk(
                        walk_length=walk_length, start_node=v, edge_type=edge_type)
                    if len(walk) < sentence_min_len or not has_higher_edge:
                        continue    
                    walks.append(walk)
        return walks

    def get_alias_edge(self, t, v, edge_type):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        # for x in G.neighbors(v):
        for x in get_neighbors_edgeType(self.G, v, edge_type):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self, edge_type):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        print("edge type : ", edge_type)
        G = self.G
        alias_nodes = {}
        cnt = 1
        node_len = len(G.nodes())
        for node in tqdm(G.nodes()):
            
            # unnormalized_probs = [G[node][nbr].get('weight', 1.0)
            #                       for nbr in G.neighbors(node)]
            # unnormalized_probs = []
            # for nbr in G.neighbors(node):
            #     if G[node][nbr].get('EdgeType') == edge_type:
            #         unnormalized_probs.append(G[node][nbr].get('weight', 1.0))

            unnormalized_probs = [G[node][nbr][0].get('weight', 1.0)
                                  for nbr in get_neighbors_edgeType(self.G, node, edge_type)]
            # if node == "{L5K1-L1NL17VJ-9414WZTP}":
            #     print("______-------amazing_________--------")
            #     print(get_neighbors_edgeType(self.G, node, edge_type))
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

            cnt += 1
        # use_rejection_sampling = 1的话 就不进入下面了
        if not self.use_rejection_sampling:
            alias_edges = {}
            print("Use reject sample")
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], edge_type)
                if not G.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0], edge_type)
                self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return


class BiasedWalker:
    def __init__(self, idx2node, temp_path):

        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path
        pass

    def simulate_walks(self, num_walks, walk_length, stay_prob=0.3, workers=1, verbose=0):

        layers_adj = pd.read_pickle(self.temp_path+'layers_adj.pkl')
        layers_alias = pd.read_pickle(self.temp_path+'layers_alias.pkl')
        layers_accept = pd.read_pickle(self.temp_path+'layers_accept.pkl')
        gamma = pd.read_pickle(self.temp_path+'gamma.pkl')
        walks = []
        initialLayer = 0

        nodes = self.idx  # list(self.g.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._exec_random_walk(layers_adj, layers_accept, layers_alias,
                                                    v, walk_length, gamma, stay_prob))
        return walks

    def _exec_random_walk(self, graphs, layers_accept, layers_alias, v, walk_length, gamma, stay_prob=0.3):
        initialLayer = 0
        layer = initialLayer

        path = []
        path.append(self.idx2node[v])

        while len(path) < walk_length:
            r = random.random()
            if(r < stay_prob):  # same layer
                v = chooseNeighbor(v, graphs, layers_alias,
                                   layers_accept, layer)
                path.append(self.idx2node[v])
            else:  # different layer
                r = random.random()
                try:
                    x = math.log(gamma[layer][v] + math.e)
                    p_moveup = (x / (x + 1))
                except:
                    print(layer, v)
                    raise ValueError()

                if(r > p_moveup):
                    if(layer > initialLayer):
                        layer = layer - 1
                else:
                    if((layer + 1) in graphs and v in graphs[layer + 1]):
                        layer = layer + 1

        return path


def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):

    v_list = graphs[layer][v]

    idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
    v = v_list[idx]

    return v

# print('1')
