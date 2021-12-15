# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)



"""
from ..walker import RandomWalker
# from .w2v import word_embedding
from gensim.models import Word2Vec
from .w2v import word_embedding
import time
import pandas as pd
import tqdm
# import pdb


class DeepWalk:
    # num_walks 100; workers = 10
    def __init__(self, graph, walk_length, num_walks, workers=1, edge_type_group=[[1], [1,2], [1,2,3]]):
        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.sentence_min_len = 5

        # 随机游走, 1/p 是不进行游走的概率，1/q是访问距离为2的节点的概率，邻接访问概率为1
        self.walker = RandomWalker(graph, p=100, q=1, )
        print("Start to Random walk ... ")
        print('prob1 start!')
        st = time.time()
        self.walker.preprocess_transition_probs(edge_type_group[0])
        print('prob1 end! cost time ', time.time() - st)

        print('sentences1 start!')
        st = time.time()
        self.sentences1 = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, sentence_min_len=self.sentence_min_len, verbose=1, edge_type=edge_type_group[0])
        print('sentences1 end! cost time ', time.time() - st)

        print('prob2 start!')
        st = time.time()
        self.walker.preprocess_transition_probs(edge_type_group[1])
        print('prob2 end! cost time ', time.time() - st)

        print('sentences2 start!')
        st = time.time()
        self.sentences2 = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, sentence_min_len=self.sentence_min_len, verbose=1, edge_type=edge_type_group[1])
        print('sentences2 end! cost time ', time.time() - st)
        # self.walker.preprocess_transition_probs(edge_type_group[2])
        # self.sentences3 = self.walker.simulate_walks(
        #     num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1, edge_type=edge_type_group[2])
        print("End to Random walk ... ")

        print('check_sentence1 start!')
        st = time.time()
        self.check_sentences(self.graph, self.sentences1, edge_type_group[0])
        print('check_sentence1 end! cost time ', time.time() - st)

        print('check_sentence2 start!')
        st = time.time()
        self.check_sentences(self.graph, self.sentences2, edge_type_group[1])
        print('check_sentence1 end! cost time ', time.time() - st)
        # self.check_sentences(self.graph, self.sentences3, edge_type_group[2])

        # self.sentences = self.sentences1[:len(self.sentences3) // 3] + self.sentences2[:len(self.sentences3) // 2] + self.sentences3
        self.sentences = self.sentences1[:len(self.sentences2) // 2] + self.sentences2

    def check_sentences(self, graph, sentecnes, edge, sentence_min_len = 5):
        print("Before checking, the number of sentences : ", len(sentecnes))
        # 过短的舍弃
        for sentence in sentecnes:
            if len(sentence) < sentence_min_len:
                sentecnes.remove(sentence)
                continue
            
            has_edge = False
            # 至少含有一个更高级规则的边
            for i in range(len(sentence) - 1):
                if graph[sentence[i]][sentence[i+1]][0]['EdgeType'] == edge[-1]:
                    has_edge = True
                    break
            if not has_edge:
                sentecnes.remove(sentence)
        print("After checking, the number of sentences : ", len(sentecnes))

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        model = word_embedding(self.sentences, window_size, embed_size, iter, batch_size=512)
        model.train()
        self.w2v_model = model
        # kwargs["sentences"] = self.sentences
        # kwargs["min_count"] = kwargs.get("min_count", 0)
        # kwargs["size"] = embed_size
        # kwargs["sg"] = 1  # skip gram
        # kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        # kwargs["workers"] = workers
        # kwargs["window"] = window_size
        # kwargs["epochs"] = iter
        
        # print("Learning embedding vectors...")
        # model = Word2Vec(**kwargs)
        # print("Learning embedding vectors done!")
        #
        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            # pdb.set_trace()
            self._embeddings[word] = self.w2v_model.wv[word]
            # self._embeddings[word] = self.w2v_model.get_embeddings(word)

        return self._embeddings
