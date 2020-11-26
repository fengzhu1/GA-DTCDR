# -*- Encoding:UTF-8 -*-

import numpy as np
import random
import networkx as nx

# Construct the graph for Grapy Embedding accroding to this samplling probability
samplling_probability=0.05
class DataSet(object):
    def __init__(self, fileName,model_D2V):
        self.model_D2V=model_D2V
        self.graph=nx.Graph()
        self.data, self.shape = self.getData(fileName)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()
        if model_D2V:
            self.adj=self.getAdj()
        else:
            self.adj=None

    def getData(self, fileName):
        print("Loading %s data set..."%(fileName))
        data = []
        filePath = './Data/'+fileName+'/ratings.dat'
        u = 0
        i = 0
        maxr = 0.0
        with open(filePath, 'r') as f:
            for line in f:
                if line:
                    lines = line.split("\t")
                    user = int(lines[0])
                    movie = int(lines[1])
                    score = float(lines[2])
                    data.append((user, movie, score, 0))
                    if user > u:
                        u = user
                    if movie > i:
                        i = movie
                    if score > maxr:
                        maxr = score
        self.maxRate = maxr
        print("Loading Success!\n"
              "Data Info:\n"
              "\tUser Num: {}\n"
              "\tItem Num: {}\n"
              "\tData Size: {}".format(u, i, len(data)))
        self.graph.add_nodes_from(range(u+i))
        return data, [u, i]

    def getTrainTest(self):
        data = self.data
        data = sorted(data, key=lambda x: (x[0], x[3]))
        train = []
        test = []
        for i in range(len(data)-1):
            user = data[i][0]-1
            item = data[i][1]-1
            rate = data[i][2]
            if data[i][0] != data[i+1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate))

        test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))
        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        for s in testData:
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u)
            tmp_item.append(i)
            neglist = set()
            neglist.add(i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (u, j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)]
    def getAdj(self):
        n_allnodes=self.shape[0]+self.shape[1] # the number of all nodes (users and items)
        adj=np.zeros([1,n_allnodes,n_allnodes], dtype=np.float32)
        # User-item interactions
        for i in self.train:
            user = i[0]
            item = i[1]
            rating= i[2]
            weight=rating/5
            adj[0][user][self.shape[0]+item]=1 #[0,self.shape[0]-1]: users, [self.shape[0],n_allnodes-1]: items
            adj[0][self.shape[0]+item][user]=1
            self.graph.add_weighted_edges_from([(user,self.shape[0]+item,weight)])
        
        for i in range(self.shape[0]):
            for j in range(i+1,self.shape[0]):
                sim=self.model_D2V.docvecs.similarity(i,j)
                rand=random.uniform(0,1)
                if rand< sim*samplling_probability:
                    adj[0][i][j]=1
                    adj[0][j][i]=1
                    self.graph.add_weighted_edges_from([(i,j,sim)])
        
        for i in range(self.shape[0],self.shape[0]+self.shape[1]):
            for j in range(i+1,self.shape[0]+self.shape[1]):
                sim=self.model_D2V.docvecs.similarity(i,j)
                rand=random.uniform(0,1)
                if rand< sim*samplling_probability:
                    adj[0][i][j]=1
                    adj[0][j][i]=1
                    self.graph.add_weighted_edges_from([(i,j,sim)])
            
        return adj
    
    def adj_to_bias(self,adj, sizes, nhood=1):
        nb_graphs = adj.shape[0]
        mt = np.empty(adj.shape)
        for g in range(nb_graphs):
            mt[g] = np.eye(adj.shape[1])
            for _ in range(nhood):
                mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
            for i in range(sizes[g]):
                for j in range(sizes[g]):
                    if mt[g][i][j] > 0.0:
                        mt[g][i][j] = 1.0
                        #print("g: {}, i: {}, j: {}".format(g,i,j))
        return -1e9 * (1.0 - mt)
        