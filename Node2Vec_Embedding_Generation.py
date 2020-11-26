# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:12:51 2018
Function Description: Generate node2vec embeddings of users and items
@author: Feng Zhu
"""
from DataSet import DataSet
from gensim.models.doc2vec import Doc2Vec
from node2vec import Node2Vec

if __name__ == '__main__':
    domains=['douban_movie','douban_book','douban_music']
    KList=[128,64,32,16,8]
    for domain in domains:
        for K_Size in KList:
            model_D2V=Doc2Vec.load("Doc2vec_"+domain+"_VSize_"+str(K_Size)+".model")
            dataSet = DataSet(domain,model_D2V)
            adj = dataSet.adj
            graph=dataSet.graph
            node2vec = Node2Vec(graph, dimensions=K_Size, walk_length=30, num_walks=100, workers=1)
            model_N2V = node2vec.fit(window=10, min_count=1, batch_words=4)
            model_N2V.save("Node2vec_"+domain+"_KSize_"+str(K_Size)+"_new.model")