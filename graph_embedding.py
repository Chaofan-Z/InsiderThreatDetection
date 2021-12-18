
import numpy as np
import os

from tqdm.std import tqdm
from graph_emb.classify import read_node_label, Classifier
from graph_emb import DeepWalk
from graph_emb.models import w2v
from graph_emb.models.w2v import word_embedding

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import pickle
import time


def get_answer(answer_dir, version):
    # r6.2
    if '6' in version or 'part' in version:
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
                with open(os.path.join(answer_dir, version+str('-')+str(la)+'/') + file_name) as file:
                    for line in file:
                        line = line.strip().split(',')
                        label[line[1].strip("\"")] = la
    # label : {node_id : label(number)}
    return label

def read_sentence(sentence_path):
    print("Read sentences")
    st = time.time()
    sentences = []
    with open(sentence_path + "/sentence_2.txt") as file:
        for line in file:
            sentences.append(line.strip().split('\t'))

    sentences2_len = len(sentences)

    with open(sentence_path + "/sentence_1.txt") as file:
        cnt = 0
        for line in file:
            if cnt < sentences2_len // 2:
                sentences.append(line.strip().split('\t'))
    print("Read sentences Done, cost : %.2f", time.time() - st)

    return sentences    

if __name__ == '__main__':

    # data_version = "r6.2"
    # data_version = "r_part"

    direct_w2v = 1
    data_dir = "/mnt/188b5285-b188-4759-81ac-763ab8cbc6bf/InsiderThreatData/"

    data_version = "r5.2"
    version = "5"
    # version = "half"

    # data_version = "r_part"
    # version = "1"

    if not os.path.exists(os.path.join("./output", data_version, version)):
        os.makedirs(os.path.join("./output", data_version, version))
    # data_version = "r_part"
    # version = "1"

    # label = get_answer(os.path.join(data_dir, "answers"), data_version)
    st = time.time()
    print("Start to read graph")
    G = nx.read_gpickle("./output/%s/%s/graph/activity_graph.gpickle"%(data_version, version))
    print("End to read graph, cost ", time.time()-st)

    if not direct_w2v:
        print("*" * 10 + " Train by DeepWalk" + "*" * 10)
        # 序列长度，xxx，并行worker数量
        # 这里正常训练流程
        model = DeepWalk(G, data_version, version, walk_length=30, num_walks=1, workers=1, edge_type_group=[[1], [1,2]])
        model.train(window_size=5, iter=50) 
        # embeddings
        # {node_id : [value1, value2, ....]}
        # embeddings = model.get_embeddings()
        # with open(os.path.join("./output", data_version, version) + "embedding.pickle", 'wb') as file:
        #     pickle.dump(embeddings, file)
        # print("End to save Embedding ...")
 
    else :
        print("*" * 10 + "Direct train by w2v" + "*" * 10)
        sentences = read_sentence(os.path.join("./output", data_version, version))
        w2v_model = word_embedding(sentences, max_window_size=5, embed_size=128, graph = G, 
            data_version=data_version, version=version, num_epochs=10, batch_size=256)
        import gc
        del G
        gc.collect()
        # print(G)
        # for node in w2v_model.graph_nodes:
        #     print(node)
        w2v_model.train()
        


