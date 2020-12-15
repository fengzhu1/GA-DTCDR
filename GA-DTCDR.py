# -*- coding: utf-8 -*-
"""
Created on Friday April 05 14:46:58 2019
Function Description: Make dual-target cross-domain recommendations (DTCDR) by using element-wise Attention mechanism to combine embeddings
Paper: A Graphical and Attentional Framework for Dual-Target Cross-Domain Recommendation (IJCAI 2020)
@author: Feng Zhu
"""
import tensorflow as tf
import numpy as np
import argparse
from DataSet import DataSet
import sys
import os
import heapq
import math
import scipy.io as scio
from gensim.models.word2vec import Word2Vec
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(dataName_A, dataName_B, K_size):
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-dataName_A',
                        action='store',
                        dest='dataName_A',
                        default=dataName_A)
    parser.add_argument('-dataName_B',
                        action='store',
                        dest='dataName_B',
                        default=dataName_B)
    parser.add_argument('-negNum',
                        action='store',
                        dest='negNum',
                        default=7,
                        type=int)
    parser.add_argument('-userLayer',
                        action='store',
                        dest='userLayer',
                        default=[
                            K_size, 2 * K_size, 4 * K_size, 8 * K_size,
                            4 * K_size, 2 * K_size, K_size
                        ])
    parser.add_argument('-itemLayer',
                        action='store',
                        dest='itemLayer',
                        default=[
                            K_size, 2 * K_size, 4 * K_size, 8 * K_size,
                            4 * K_size, 2 * K_size, K_size
                        ])
    parser.add_argument('-KSize', action='store', dest='KSize', default=K_size)
    parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lambdad',
                        action='store',
                        dest='lambdad',
                        default=0.001)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001)
    parser.add_argument('-maxEpochs',
                        action='store',
                        dest='maxEpochs',
                        default=50,
                        type=int)
    parser.add_argument('-batchSize',
                        action='store',
                        dest='batchSize',
                        default=4096,
                        type=int)
    parser.add_argument('-earlyStop',
                        action='store',
                        dest='earlyStop',
                        default=5)
    parser.add_argument('-checkPoint',
                        action='store',
                        dest='checkPoint',
                        default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=10)
    args = parser.parse_args()

    classifier = Model(args)

    classifier.run()


class Model:
    def __init__(self, args):
        self.dataName_A = args.dataName_A
        self.dataName_B = args.dataName_B
        self.KSize = args.KSize
        self.model_N2V_A = Word2Vec.load("Node2vec_" + self.dataName_A +
                                         "_KSize_" + str(self.KSize) +
                                         ".model")
        self.model_N2V_B = Word2Vec.load("Node2vec_" + self.dataName_B +
                                         "_KSize_" + str(self.KSize) +
                                         ".model")
        self.dataSet_A = DataSet(self.dataName_A, None)
        self.dataSet_B = DataSet(self.dataName_B, None)
        self.shape_A = self.dataSet_A.shape
        self.maxRate_A = self.dataSet_A.maxRate
        self.shape_B = self.dataSet_B.shape
        self.maxRate_B = self.dataSet_B.maxRate
        self.train_A = self.dataSet_A.train
        self.test_A = self.dataSet_A.test

        self.train_B = self.dataSet_B.train
        self.test_B = self.dataSet_B.test

        self.negNum = args.negNum
        self.testNeg_A = self.dataSet_A.getTestNeg(self.test_A, 99)
        self.testNeg_B = self.dataSet_B.getTestNeg(self.test_B, 99)
        self.add_embedding_matrix()

        self.add_placeholders()

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.add_model()
        self.lambdad = args.lambdad
        self.add_loss()

        self.lr = args.lr
        self.add_train_step()

        self.checkPoint = args.checkPoint
        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop

    def add_placeholders(self):
        self.user_A = tf.placeholder(tf.int32)
        self.item_A = tf.placeholder(tf.int32)
        self.rate_A = tf.placeholder(tf.float32)
        self.drop_A = tf.placeholder(tf.float32)
        self.user_B = tf.placeholder(tf.int32)
        self.item_B = tf.placeholder(tf.int32)
        self.rate_B = tf.placeholder(tf.float32)
        self.drop_B = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_item_embedding_A = tf.convert_to_tensor(
            self.dataSet_A.getEmbedding())
        self.item_user_embedding_A = tf.transpose(self.user_item_embedding_A)
        self.user_item_embedding_B = tf.convert_to_tensor(
            self.dataSet_B.getEmbedding())
        self.item_user_embedding_B = tf.transpose(self.user_item_embedding_B)

    def add_model(self):
        user_input_A = tf.nn.embedding_lookup(self.user_item_embedding_A,
                                              self.user_A)
        item_input_A = tf.nn.embedding_lookup(self.item_user_embedding_A,
                                              self.item_A)
        user_input_B = tf.nn.embedding_lookup(self.user_item_embedding_B,
                                              self.user_B)
        item_input_B = tf.nn.embedding_lookup(self.item_user_embedding_B,
                                              self.item_B)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape,
                                                   dtype=tf.float32,
                                                   stddev=0.01),
                               name=name)

        with tf.name_scope("User_Layer"):
            # Choose the node2vec embeddings as the input
            node_features_A = self.model_N2V_A.wv.vectors
            node_features_B = self.model_N2V_B.wv.vectors
            user_input_A = tf.nn.embedding_lookup(node_features_A, self.user_A)
            user_W1_A = init_variable([self.KSize, self.userLayer[0]],
                                      "user_W1_A")
            user_out_A = tf.matmul(user_input_A, user_W1_A)
            user_input_B = tf.nn.embedding_lookup(node_features_B, self.user_B)
            user_W1_B = init_variable([self.KSize, self.userLayer[0]],
                                      "user_W1_B")
            user_out_B = tf.matmul(user_input_B, user_W1_B)

            # Element-wise Attention for common users
            user_W_Attention_A_A = tf.Variable(
                tf.truncated_normal(shape=[self.shape_A[1], self.userLayer[0]],
                                    dtype=tf.float32,
                                    stddev=0.01),
                name="user_W_Attention_A_A"
            )  # the weights of Domain A for Domain A
            user_W_Attention_B_A = 1 - user_W_Attention_A_A  # the weights of Domain B for Domain A
            user_W_Attention_A_A_lookup = tf.nn.embedding_lookup(
                user_W_Attention_A_A, self.user_A)
            user_W_Attention_B_A_lookup = tf.nn.embedding_lookup(
                user_W_Attention_B_A, self.user_B)
            user_out_A_Combined = tf.add(
                tf.multiply(user_out_A, user_W_Attention_A_A_lookup),
                tf.multiply(user_out_B, user_W_Attention_B_A_lookup))
            user_W_Attention_B_B = tf.Variable(
                tf.truncated_normal(shape=[self.shape_B[1], self.userLayer[0]],
                                    dtype=tf.float32,
                                    stddev=0.01),
                name="user_W_Attention_B_B"
            )  # the weights of Domain B for Domain B
            user_W_Attention_A_B = 1 - user_W_Attention_B_B  # the weights of Domain A for Domain B
            user_W_Attention_B_B_lookup = tf.nn.embedding_lookup(
                user_W_Attention_B_B, self.user_B)
            user_W_Attention_A_B_lookup = tf.nn.embedding_lookup(
                user_W_Attention_A_B, self.user_A)
            user_out_B_Combined = tf.add(
                tf.multiply(user_out_A, user_W_Attention_A_B_lookup),
                tf.multiply(user_out_B, user_W_Attention_B_B_lookup))
            user_out_A = user_out_A_Combined
            user_out_B = user_out_B_Combined

            # full-connected layers (MLP)
            for i in range(0, len(self.userLayer) - 1):
                W_A = init_variable([self.userLayer[i], self.userLayer[i + 1]],
                                    "user_W_A" + str(i + 2))
                b_A = init_variable([self.userLayer[i + 1]],
                                    "user_b_A" + str(i + 2))
                user_out_A = tf.nn.relu(tf.add(tf.matmul(user_out_A, W_A),
                                               b_A))
                W_B = init_variable([self.userLayer[i], self.userLayer[i + 1]],
                                    "user_W_B" + str(i + 2))
                b_B = init_variable([self.userLayer[i + 1]],
                                    "user_b_B" + str(i + 2))
                user_out_B = tf.nn.relu(tf.add(tf.matmul(user_out_B, W_B),
                                               b_B))

        with tf.name_scope("Item_Layer"):
            # Choose the node2vec embeddings as the input
            node_features_A = self.model_N2V_A.wv.vectors
            node_features_B = self.model_N2V_B.wv.vectors
            item_input_A = tf.nn.embedding_lookup(
                node_features_A, self.shape_A[0] + self.item_A)
            item_W1_A = init_variable([self.KSize, self.itemLayer[0]],
                                      "item_W1_A")
            item_out_A = tf.matmul(item_input_A, item_W1_A)
            item_input_B = tf.nn.embedding_lookup(
                node_features_B, self.shape_B[0] + self.item_B)
            item_W1_B = init_variable([self.KSize, self.itemLayer[0]],
                                      "item_W1_B")
            item_out_B = tf.matmul(item_input_B, item_W1_B)
            # full-connected layers (MLP)
            for i in range(0, len(self.itemLayer) - 1):
                W_A = init_variable([self.itemLayer[i], self.itemLayer[i + 1]],
                                    "item_W_A" + str(i + 2))
                b_A = init_variable([self.itemLayer[i + 1]],
                                    "item_b_A" + str(i + 2))
                item_out_A = tf.nn.relu(tf.add(tf.matmul(item_out_A, W_A),
                                               b_A))
                W_B = init_variable([self.itemLayer[i], self.itemLayer[i + 1]],
                                    "item_W_B" + str(i + 2))
                b_B = init_variable([self.itemLayer[i + 1]],
                                    "item_b_B" + str(i + 2))
                item_out_B = tf.nn.relu(tf.add(tf.matmul(item_out_B, W_B),
                                               b_B))

        norm_user_output_A = tf.sqrt(
            tf.reduce_sum(tf.square(user_out_A), axis=1))
        norm_item_output_A = tf.sqrt(
            tf.reduce_sum(tf.square(item_out_A), axis=1))
        norm_user_output_B = tf.sqrt(
            tf.reduce_sum(tf.square(user_out_B), axis=1))
        norm_item_output_B = tf.sqrt(
            tf.reduce_sum(tf.square(item_out_B), axis=1))
        self.regularizer_A = tf.nn.l2_loss(user_input_A) + tf.nn.l2_loss(
            item_out_A)
        self.regularizer_B = tf.nn.l2_loss(user_input_B) + tf.nn.l2_loss(
            item_out_B)
        self.y_A = tf.reduce_sum(
            tf.multiply(user_out_A, item_out_A), axis=1,
            keepdims=False) / (norm_item_output_A * norm_user_output_A)
        self.y_A = tf.maximum(1e-6, self.y_A)
        self.y_B = tf.reduce_sum(
            tf.multiply(user_out_B, item_out_B), axis=1,
            keepdims=False) / (norm_item_output_B * norm_user_output_B)
        self.y_B = tf.maximum(1e-6, self.y_B)

    def add_loss(self):
        regRate_A = self.rate_A / self.maxRate_A
        losses_A = regRate_A * tf.log(
            self.y_A) + (1 - regRate_A) * tf.log(1 - self.y_A)
        loss_A = -tf.reduce_sum(losses_A)
        self.loss_A = loss_A + self.lambdad * self.regularizer_A

        regRate_B = self.rate_B / self.maxRate_B
        losses_B = regRate_B * tf.log(
            self.y_B) + (1 - regRate_B) * tf.log(1 - self.y_B)
        loss_B = -tf.reduce_sum(losses_B)
        self.loss_B = loss_B + self.lambdad * self.regularizer_B

    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step_A = optimizer.minimize(self.loss_A)
        self.train_step_B = optimizer.minimize(self.loss_B)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def run(self):
        best_hr_A = -1
        best_NDCG_A = -1
        best_epoch_A = -1
        best_hr_B = -1
        best_NDCG_B = -1
        best_epoch_B = -1
        allResults_A = []
        allResults_B = []
        print("Start Training!")

        for epoch in range(self.maxEpochs):
            print("=" * 20 + "Epoch ", epoch, "=" * 20)
            self.run_epoch(self.sess)
            print('=' * 50)
            print("Start Evaluation!")
            topK = 10
            hr_A, NDCG_A, hr_B, NDCG_B = self.evaluate(self.sess, topK)
            allResults_A.append([epoch, topK, hr_A, NDCG_A])
            allResults_B.append([epoch, topK, hr_B, NDCG_B])
            print(
                "Epoch ", epoch,
                "Domain A: {} TopK: {} HR: {}, NDCG: {}".format(
                    self.dataName_A, topK, hr_A, NDCG_A))
            print(
                "Epoch ", epoch,
                "Domain B: {} TopK: {} HR: {}, NDCG: {}".format(
                    self.dataName_B, topK, hr_B, NDCG_B))
            if hr_A > best_hr_A:
                best_hr_A = hr_A
                best_epoch_A = epoch
            if NDCG_A > best_NDCG_A:
                best_NDCG_A = NDCG_A
            if hr_B > best_hr_B:
                best_hr_B = hr_B
                best_epoch_B = epoch
            if NDCG_B > best_NDCG_B:
                best_NDCG_B = NDCG_B
            print("=" * 20 + "Epoch ", epoch, "End" + "=" * 20)
        print(
            "Domain A: Best hr: {}, NDCG: {}, At Epoch {}; Domain B: Best hr: {}, NDCG: {}, At Epoch {}"
            .format(best_hr_A, best_NDCG_A, best_epoch_A, best_hr_B,
                    best_NDCG_B, best_epoch_B))
        bestPerformance = [[best_hr_A, best_NDCG_A, best_epoch_A],
                           [best_hr_B, best_NDCG_B, best_epoch_B]]
        # I save the experimental result in the form of mat, it is easy to draw a figure in Matlab.
        matname = 'GA-DTCDR_' + str(self.dataName_A) + '_' + str(
            self.dataName_B) + '_KSize_' + str(self.KSize) + '_Result.mat'
        scio.savemat(
            matname, {
                'allResults_A': allResults_A,
                'allResults_B': allResults_B,
                'bestPerformance': bestPerformance
            })
        print("Training complete!")

    def run_epoch(self, sess, verbose=10):
        train_u_A, train_i_A, train_r_A = self.dataSet_A.getInstances(
            self.train_A, self.negNum)
        train_len_A = len(train_u_A)
        shuffled_idx_A = np.random.permutation(np.arange(train_len_A))
        train_u_A = train_u_A[shuffled_idx_A]
        train_i_A = train_i_A[shuffled_idx_A]
        train_r_A = train_r_A[shuffled_idx_A]

        train_u_B, train_i_B, train_r_B = self.dataSet_B.getInstances(
            self.train_B, self.negNum)
        train_len_B = len(train_u_B)
        shuffled_idx_B = np.random.permutation(np.arange(train_len_B))
        train_u_B = train_u_B[shuffled_idx_B]
        train_i_B = train_i_B[shuffled_idx_B]
        train_r_B = train_r_B[shuffled_idx_B]

        num_batches_A = len(train_u_A) // self.batchSize + 1
        num_batches_B = len(train_u_B) // self.batchSize + 1

        losses_A = []
        losses_B = []
        max_num_batches = max(num_batches_A, num_batches_B)
        for i in range(max_num_batches):
            min_idx = i * self.batchSize
            max_idx_A = np.min([train_len_A, (i + 1) * self.batchSize])
            max_idx_B = np.min([train_len_B, (i + 1) * self.batchSize])
            if min_idx < train_len_A:  # the training for domain A has not completed
                train_u_batch_A = train_u_A[min_idx:max_idx_A]
                train_i_batch_A = train_i_A[min_idx:max_idx_A]
                train_r_batch_A = train_r_A[min_idx:max_idx_A]
                feed_dict_A = self.create_feed_dict(train_u_batch_A,
                                                    train_i_batch_A, 'A',
                                                    train_r_batch_A)
                _, tmp_loss_A, _y_A = sess.run(
                    [self.train_step_A, self.loss_A, self.y_A],
                    feed_dict=feed_dict_A)
                losses_A.append(tmp_loss_A)
            if min_idx < train_len_B:  # the training for domain B has not completed
                train_u_batch_B = train_u_B[min_idx:max_idx_B]
                train_i_batch_B = train_i_B[min_idx:max_idx_B]
                train_r_batch_B = train_r_B[min_idx:max_idx_B]
                feed_dict_B = self.create_feed_dict(train_u_batch_B,
                                                    train_i_batch_B, 'B',
                                                    train_r_batch_B)
                _, tmp_loss_B, _y_B = sess.run(
                    [self.train_step_B, self.loss_B, self.y_B],
                    feed_dict=feed_dict_B)
                losses_B.append(tmp_loss_B)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {};'.format(
                    i, num_batches_A, np.mean(losses_A[-verbose:])))
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches_B, np.mean(losses_B[-verbose:])))
                sys.stdout.flush()
        loss_A = np.mean(losses_A)
        loss_B = np.mean(losses_B)
        print("\nMean loss in this epoch is: Domain A={};Domain B={}".format(
            loss_A, loss_B))
        return (loss_A, loss_B)

    def create_feed_dict(self, u, i, dataset, r=None, drop=None):
        if dataset == 'A':
            return {
                self.user_A: u,
                self.item_A: i,
                self.rate_A: r,
                self.drop_A: drop,
                self.user_B: u,
                self.item_B: [],
                self.rate_B: [],
                self.drop_B: drop
            }
        else:
            return {
                self.user_B: u,
                self.item_B: i,
                self.rate_B: r,
                self.drop_B: drop,
                self.user_A: u,
                self.item_A: [],
                self.rate_A: [],
                self.drop_A: drop
            }

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0

        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i + 2)
            return 0

        hr_A = []
        NDCG_A = []
        testUser_A = self.testNeg_A[0]
        testItem_A = self.testNeg_A[1]
        hr_B = []
        NDCG_B = []
        testUser_B = self.testNeg_B[0]
        testItem_B = self.testNeg_B[1]
        for i in range(len(testUser_A)):
            target = testItem_A[i][0]
            feed_dict_A = self.create_feed_dict(testUser_A[i], testItem_A[i],
                                                'A')
            predict_A = sess.run(self.y_A, feed_dict=feed_dict_A)

            item_score_dict = {}

            for j in range(len(testItem_A[i])):
                item = testItem_A[i][j]
                item_score_dict[item] = predict_A[j]

            ranklist = heapq.nlargest(topK,
                                      item_score_dict,
                                      key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr_A.append(tmp_hr)
            NDCG_A.append(tmp_NDCG)
        for i in range(len(testUser_B)):
            target = testItem_B[i][0]
            feed_dict_B = self.create_feed_dict(testUser_B[i], testItem_B[i],
                                                'B')
            predict_B = sess.run(self.y_B, feed_dict=feed_dict_B)

            item_score_dict = {}

            for j in range(len(testItem_B[i])):
                item = testItem_B[i][j]
                item_score_dict[item] = predict_B[j]

            ranklist = heapq.nlargest(topK,
                                      item_score_dict,
                                      key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr_B.append(tmp_hr)
            NDCG_B.append(tmp_NDCG)
        return np.mean(hr_A), np.mean(NDCG_A), np.mean(hr_B), np.mean(NDCG_B)


if __name__ == '__main__':
    tasks = [['douban_movie', 'douban_book'], ['douban_movie', 'douban_music']]
    KList = [32, 128, 64, 16, 8]
    for K_Size in KList:
        for [domain_A, domain_B] in tasks:
            main(domain_A, domain_B, K_Size)
