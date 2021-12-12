
import numpy as np
from graph_emb.classify import read_node_label, Classifier
from graph_emb import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

G = nx.read_gpickle("./our_data/activity_graph.gpickle")

# 序列长度，xxx，并行worker数量
model = DeepWalk(G, walk_length=10, num_walks=80, workers=1, edge_type=[1])
model.train(window_size=5, iter=3) 
embeddings = model.get_embeddings()

train_X = []
train_X_id = []

for k, v in embeddings.items():
    train_X.append(v)
    train_X_id.append(v)

train_X = np.array(train_X)
clustering = DBSCAN().fit(train_X)
# evaluate_embeddings(embeddings)
# plot_embeddings(embeddings)
