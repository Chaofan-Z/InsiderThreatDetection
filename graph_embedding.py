
import numpy as np
import os
from graph_emb.classify import read_node_label, Classifier
from graph_emb import DeepWalk
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import pickle


def get_answer(answer_dir, version):
    # r6.2
    if '6' in version:
        label_num = [i for i in range(1, 6)]
        label = {}
        for la in label_num:
            with open(os.path.join(answer_dir, version) + '-' + str(la) + '.csv') as file:
                for line in file:
                    line = line.strip().split(',')
                    label[line[1].strip("\"")] = la
    # r5.2
    else :
        label_num = [i for i in range(1, 5)]
        label = {}
        for la in label_num:
            for file_name in os.listdir(os.path.join(answer_dir, version+str('-')+str(la))):
                print(file_name)
                with open(file_name) as file:
                    for line in file:
                        line = line.strip().split(',')
                        label[line[1].strip("\"")] = la
    # label : {node_id : label(number)}
    return label

if __name__ == '__main__':


    # data_version = "r6.2"
    # data_version = "r_part"

    data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"
    data_version = "r5.2"
    data_version = "r_part"

    version = "1"

    # label = get_answer(os.path.join(data_dir, "answers"), data_version)

    G = nx.read_gpickle("./our_data/activity_graph.gpickle")
    # 序列长度，xxx，并行worker数量

    model = DeepWalk(G, walk_length=10, num_walks=100, workers=10, edge_type_group=[[1], [1,2], [1,2,3]])
    model.train(window_size=5, iter=3) 
    # embeddings = model.get_embeddings()
    
    # if not os.path.exists(os.path.join("./output", data_version, version)):
    #     os.makedirs(os.path.join("./output", data_version, version))
    # with open(os.path.join("./output", data_version, version) + "embedding.pickle", 'wb') as file:
    #     pickle.dump(embeddings, file)


