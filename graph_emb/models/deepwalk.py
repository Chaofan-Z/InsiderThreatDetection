# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)



"""
from ..walker import RandomWalker
from gensim.models import Word2Vec
import pandas as pd
import pdb


class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1, edge_type_group=[[1], [1,2], [1,2,3]]):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        # 随机游走, 1/p 是不进行游走的概率，1/q是访问距离为2的节点的概率，邻接访问概率为1
        self.walker = RandomWalker(graph, p=100, q=1, )

        self.walker.preprocess_transition_probs(edge_type_group[0])
        self.sentences1 = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1, edge_type=edge_type_group[0])
        
        self.walker.preprocess_transition_probs(edge_type_group[1])
        self.sentences2 = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1, edge_type=edge_type_group[1])
        
        self.walker.preprocess_transition_probs(edge_type_group[2])
        self.sentences3 = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1, edge_type=edge_type_group[2])
        
        # print(len(self.sentences1))
        # print(self.sentences1[:2])
        # print(len(self.sentences2))
        # print(self.sentences2[:2])
        # print(len(self.sentences3))
        # print(self.sentences3[:2])
        # print(len(self.sentences))
        # print(self.sentences[0])

        self.sentences = self.sentences1 + self.sentences2 + self.sentences3

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        print("Learning embedding vectors done!")

        self.w2v_model = model
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            # pdb.set_trace()
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
