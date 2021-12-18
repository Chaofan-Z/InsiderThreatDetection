# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:03:17 2021

@author: 李天赐
"""

import collections
import math
import random
import sys
import time
import os
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import pdb
import datetime
from tqdm import tqdm
import pickle


def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, list中的每个元素都是__getitem__得到的结果"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)


def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):  # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)


def sigmd(x):
    return - math.log(1 / (1 + math.exp(-x)))


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    # pdb.set_trace()
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class word_embedding:
    def __init__(self, raw_dataset, max_window_size, embed_size, graph, data_version, version, num_epochs=10, batch_size=5, lr=0.01):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_nodes = graph.nodes()

        # print(graph)

        self.all_model_dir = os.path.join("./output", data_version, version, "w2vmodel")
        if not os.path.exists(self.all_model_dir):
            os.makedirs(self.all_model_dir)
        
        self.data_version, self.version = data_version, version

        counter = collections.Counter([tk for st in raw_dataset for tk in st])
        counter = dict(counter.items())
        self.idx_to_token = [tk for tk, _ in counter.items()]
        self.token_to_idx = {tk: idx for idx, tk in enumerate(self.idx_to_token)}
        dataset = [[self.token_to_idx[tk] for tk in st if tk in self.token_to_idx]
                   for st in raw_dataset]
        print("Data preprocess done..")
        self.max_window_size = max_window_size
        self.embed_size = embed_size
        self.num_epochs = num_epochs

        self.net = nn.Sequential(
            nn.Embedding(num_embeddings=len(self.idx_to_token), embedding_dim=embed_size),
            nn.Embedding(num_embeddings=len(self.idx_to_token), embedding_dim=embed_size)
        )

        self.lr = lr

        num_tokens = sum([len(st) for st in dataset])

        def discard(idx):
            return random.uniform(0, 1) < 1 - math.sqrt(
                1e-4 / counter[self.idx_to_token[idx]] * num_tokens)

        subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
        all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, max_window_size)

        sampling_weights = [counter[w] ** 0.75 for w in self.idx_to_token]
        all_negatives = get_negatives(all_contexts, sampling_weights, 5)

        self.dataset = MyDataset(all_centers,
                                 all_contexts,
                                 all_negatives)

        self.batch_size = batch_size
        # num_workers = 0 if sys.platform.startswith('win32') else 4
        num_workers = 0
        self.data_iter = Data.DataLoader(self.dataset, self.batch_size, collate_fn=batchify, num_workers=num_workers)

        self.loss = SigmoidBinaryCrossEntropyLoss()

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("train on", device)
        net = self.net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        best_l_sum = 99999
        for epoch in range(self.num_epochs):
            start, l_sum, n = time.time(), 0.0, 0
            for batch in tqdm(self.data_iter):
                center, context_negative, mask, label = [d.to(device) for d in batch]

                pred = skip_gram(center, context_negative, net[0], net[1])
                # pdb.set_trace()
                # 使用掩码变量mask来避免填充项对损失函数计算的影响
                l = (self.loss(pred.view(label.shape), label, mask) *
                     mask.shape[1] / mask.float().sum(dim=1)).mean()  # 一个batch的平均loss
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                l_sum += l.cpu().item()
                n += 1

            print('epoch %d, loss %.2f, time %.2fs'
                  % (epoch + 1, l_sum / n, time.time() - start))

            if best_l_sum > l_sum:
                if epoch % 2 == 0:
                    print("---Start to save Embedding ...")
                    self._embeddings = {}
                    st = time.time()
                    for word in tqdm(self.graph_nodes):
                        # pdb.set_trace()
                        # self._embeddings[word] = self.w2v_model.wv[word]
                        self._embeddings[word] = self.get_embeddings(word)
                    with open(os.path.join("./output", self.data_version, self.version) + f"/embedding-{epoch}.pickle", 'wb') as file:
                        pickle.dump(self._embeddings, file)
                    print("---End to save Embedding ... cost : ", time.time() - st)

                time_version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                model_path = self.all_model_dir + f'/{time_version}-{self.data_version}-{self.version}-epoch{epoch}.models'
                print("---Save model to ", model_path)
                def dump_pickle(obj,file_path):
                    pickle.dump(obj,open(file_path,'wb'),protocol=4)
                dump_pickle(self.net, model_path)

    def get_embeddings(self, word):
        if word in self.token_to_idx:
            idx = self.token_to_idx[word]
            # print(self.net[0].weight[idx])
            return self.net[0].weight[idx]
        return torch.Tensor([0] * self.embed_size).to(self.device)
