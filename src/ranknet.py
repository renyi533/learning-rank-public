import os
#import ipdb
import time
import models
import random
import pickle
import numpy as np
import argparse
import tensorflow as tf
import collections
from toy_ndcg import ndcg
from ranking_utils import calc_err
from convert_data_to_np_features import *
import math


class RankNetTrainer:
    def __init__(self, n_hidden, train_relevance_labels, train_query_ids, train_features, test_relevance_labels,
                 test_query_ids, test_features, vali_relevance_labels, vali_query_ids, vali_features, model_dir):
        self.train_query_ids = train_query_ids
        labels_min = 0 #np.amin(train_relevance_labels)
        self.train_relevance_labels = train_relevance_labels - labels_min

        self.train_features = train_features
        if train_query_ids is not None:
            self.train_unique_query_ids = np.unique(self.train_query_ids)

        self.vali_query_ids = vali_query_ids
        self.vali_relevance_labels = vali_relevance_labels
        self.vali_features = vali_features
        if vali_query_ids is not None:
            self.vali_unique_query_ids = np.unique(self.vali_query_ids)

        self.test_query_ids = test_query_ids
        self.test_relevance_labels = test_relevance_labels
        self.test_features = test_features
        if test_query_ids is not None:
            self.test_unique_query_ids = np.unique(self.test_query_ids)


        self.unique_ids = np.unique(train_query_ids)
        np.random.shuffle(self.unique_ids)

        self.models_directory = model_dir
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
        self.n_hidden = n_hidden
        self.best_cost = float('inf')
        self.best_ndcg = 0.0
        self.all_costs = list()
        self.all_ndcg_scores = list()
        self.all_full_ndcg_scores = list()
        self.all_err_scores = list()
        self.all_validation_costs = list()
        self.all_validation_ndcg_scores = list()
        self.all_validation_full_ndcg_scores = list()
        self.all_validation_err_scores = list()

    def train(self, learning_rate, n_layers, lambdarank, factorized, n_features, epoch):
        x = tf.placeholder("float", [None, n_features])
        relevance_scores = tf.placeholder("float", [None, 1])
        sorted_relevance_scores = tf.placeholder("float", [None, 1])
        index_range = tf.placeholder("float", [None, 1])
        lr = tf.placeholder("float", [])
        query_indices = tf.placeholder("float", [None])
        self.learning_rate = learning_rate
        self.start_time = time.time()
        if lambdarank:
            self.filename = 'nn_lambdarank_%slayers_%shidden_lr%s' % (n_layers, self.n_hidden, ('%.0E' % self.learning_rate).replace('-', '_'))
            cost, optimizer, score = models.lambdarank_deep(x, relevance_scores, sorted_relevance_scores, index_range,
                                                        self.learning_rate, self.n_hidden, n_layers, n_features)
        elif not factorized:
            self.filename = 'nn_unfactorized_ranknet_%slayers_%shidden_lr%s' % (n_layers, self.n_hidden, ('%.0E' % self.learning_rate).replace('-', '_'))
            cost, optimizer, score = models.default_ranknet(x, relevance_scores, self.learning_rate, self.n_hidden, n_layers, n_features)
        elif factorized:
            self.filename = 'nn_factorized_ranknet_%slayers_%shidden_lr%s' % (n_layers, self.n_hidden, ('%.0E' % self.learning_rate).replace('-', '_'))
            cost, optimizer, score = models.deep_factorized_ranknet(x, relevance_scores, self.learning_rate, self.n_hidden, n_layers, n_features)
        else:
            raise('Need to specify if this model should be unfactorized, factorized, or use lambdarank!')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            ckpt = tf.train.get_checkpoint_state(self.models_directory)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path+". Will load saved model")
                saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
            else:
                print('no valid saved model found')
            c_iter = 0
            while c_iter<epoch:

                #indices = np.random.randint(1, len(self.train_features), batch_size)
                for index in range(len(self.unique_ids)):
                    c_id = self.unique_ids[index]
                    indices = np.where(self.train_query_ids == c_id)[0]

                    if lambdarank:
                        optimizer(sess, {
                            x: np.array(self.train_features[indices], ndmin=2),
                            relevance_scores: np.array(self.train_relevance_labels[indices], ndmin=2).T,
                            lr: self.learning_rate,
                            query_indices: indices,
                            index_range: np.array([float(i) for i in range(0,len(indices))], ndmin=2).T,
                            sorted_relevance_scores: np.sort(np.array(self.train_relevance_labels[indices], ndmin=2)).T[::-1]
                        })
                    else:
                        optimizer(sess, {
                            x: np.array(self.train_features[indices], ndmin=2),
                            relevance_scores: np.array(self.train_relevance_labels[indices], ndmin=2).T,
                            lr: self.learning_rate,
                            query_indices: indices
                        })
                if c_iter % 1 == 0:
                    self.check_progress(sess, saver, cost, score, x, relevance_scores, c_iter, True)
                c_iter += 1
            if self.test_features is not None:
                test_avg_cost, test_avg_err, test_avg_ndcg, test_avg_full_ndcg = self.check_scores(cost,
                  self.test_features,
                  self.test_query_ids,
                  self.test_relevance_labels,
                  relevance_scores, score, sess,
                  self.test_unique_query_ids, x)
                print('Test Cost: {:10f} NDCG: {:9f} ({:9f}) ERR: {:9f}  {:9f} s'.format(
                        test_avg_cost, test_avg_ndcg, test_avg_full_ndcg, test_avg_err, time.time() - self.start_time))
                predictions = self.compute_predictions(self.test_features, self.test_relevance_labels, relevance_scores, score, sess, x)
                filename = os.path.join(self.models_directory, 'pred.csv')
                with open(filename, 'w') as f:
                    for elem in predictions:
                        f.write(str(elem[0])+'\n')





    def check_progress(self, sess, saver, cost, score, x, relevance_scores, c_iter, save_data=True):
        train_avg_cost, train_avg_err, train_avg_ndcg, train_avg_full_ndcg = self.check_scores(cost,
            self.train_features,
            self.train_query_ids,
            self.train_relevance_labels,
            relevance_scores, score, sess,
            self.train_unique_query_ids, x)

        if self.vali_features is not None:
            vali_avg_cost, vali_avg_err, vali_avg_ndcg, vali_avg_full_ndcg = self.check_scores(cost,
              self.vali_features,
              self.vali_query_ids,
              self.vali_relevance_labels,
              relevance_scores, score, sess,
              self.vali_unique_query_ids, x)
        else:
            vali_avg_cost = vali_avg_err = vali_avg_ndcg = vali_avg_full_ndcg = float('nan')
        print('{} -- Train Cost: {:10f} NDCG: {:9f} ({:9f}) ERR: {:9f}  -- Validation Cost: {:10f} NDCG: {:9f} ({:9f}) ERR: {:9f} -- {:9f} s'.format(
            c_iter, train_avg_cost, train_avg_ndcg, train_avg_full_ndcg, train_avg_err, vali_avg_cost, vali_avg_ndcg, vali_avg_full_ndcg, vali_avg_err, time.time() - self.start_time))
        self.all_costs.append(train_avg_cost)
        self.all_full_ndcg_scores.append(train_avg_full_ndcg)
        self.all_ndcg_scores.append(train_avg_ndcg)
        self.all_err_scores.append(train_avg_err)
        self.all_validation_costs.append(vali_avg_cost)
        self.all_validation_full_ndcg_scores.append(vali_avg_full_ndcg)
        self.all_validation_ndcg_scores.append(vali_avg_ndcg)
        self.all_validation_err_scores.append(vali_avg_err)
        if self.all_validation_costs[-1] < self.best_cost:
            self.best_cost = self.all_validation_costs[-1]
            saver.save(sess, os.path.join(self.models_directory, self.filename + '_best_validation_cost'))
        if self.all_ndcg_scores[-1] > self.best_ndcg:
            self.best_ndcg = self.all_ndcg_scores[-1]
            saver.save(sess, os.path.join(self.models_directory, self.filename + '_best_train_ndcg'))
        if save_data:
            saver.save(sess, os.path.join(self.models_directory, self.filename + '_most_recent'))
            pickle.dump(self.all_costs, open(os.path.join(self.models_directory, self.filename + '_costs.p'), 'wb'))
            pickle.dump(self.all_ndcg_scores, open(os.path.join(self.models_directory, self.filename + '_ndcg_scores.p'), 'wb'))
            pickle.dump(self.all_full_ndcg_scores, open(os.path.join(self.models_directory, self.filename + '_full_ndcg_scores.p'), 'wb'))
            pickle.dump(self.all_err_scores, open(os.path.join(self.models_directory, self.filename + '_err_scores.p'), 'wb'))
            pickle.dump(self.all_validation_costs, open(os.path.join(self.models_directory, self.filename + '_validation_costs.p'), 'wb'))
            pickle.dump(self.all_validation_ndcg_scores, open(os.path.join(self.models_directory, self.filename + '_validation_ndcg_scores.p'), 'wb'))
            pickle.dump(self.all_validation_full_ndcg_scores, open(os.path.join(self.models_directory, self.filename + '_validation_full_ndcg_scores.p'), 'wb'))
            pickle.dump(self.all_validation_err_scores, open(os.path.join(self.models_directory, self.filename + '_validation_err_scores.p'), 'wb'))
        return train_avg_cost, train_avg_ndcg, train_avg_err, vali_avg_cost, vali_avg_err, vali_avg_ndcg


    def check_scores(self, cost, features, query_ids, relevance_labels, relevance_scores, score, sess,
                     unique_query_ids, x):
        costs = list()
        ndcg_scores = list()
        full_ndcg_scores = list()
        err_scores = list()
        assert len(unique_query_ids) > 0
        for c_id in unique_query_ids:
            query_indices = np.where(query_ids == c_id)[0]
            c_cost = sess.run(cost, feed_dict={
                x: np.array(features[query_indices], ndmin=2),
                relevance_scores: np.array(relevance_labels[query_indices], ndmin=2).T })
            predicted_score = score(sess, {
                x: np.array(features[query_indices], ndmin=2),
                relevance_scores: np.array(relevance_labels[query_indices], ndmin=2).T })
            pred_query_type = np.dtype(
                [('predicted_scores', predicted_score.dtype),
                 ('query_int', query_indices.dtype)])
            pred_query = np.empty(len(predicted_score), dtype=pred_query_type)
            pred_query['predicted_scores'] = np.reshape(predicted_score, [-1])
            pred_query['query_int'] = query_indices
            scored_pred_query = np.sort(pred_query, order='predicted_scores')[::-1]

            costs.append(c_cost)
            ndcg_scores.append(ndcg(relevance_labels[scored_pred_query['query_int']]))
            full_ndcg_scores.append(ndcg(relevance_labels[scored_pred_query['query_int']], top_ten=False))
            err_scores.append(calc_err(relevance_labels[scored_pred_query['query_int']]))
        avg_cost = sum(costs) / len(costs)
        avg_ndcg = np.mean(np.array(ndcg_scores))
        avg_full_ndcg = np.mean(np.array(full_ndcg_scores))
        avg_err = np.mean(np.array(err_scores))
        return avg_cost, avg_err, avg_ndcg, avg_full_ndcg

    def compute_predictions(self, features, relevance_labels, relevance_scores, score, sess, x):
        predicted_score = score(sess, {
            x: np.array(features, ndmin=2),
            relevance_scores: np.array(relevance_labels, ndmin=2).T })
        return predicted_score

def load_files(data_dir):
    if os.path.exists(data_dir):
        relevance_labels = np.load(os.path.join(data_dir, LABEL_LIST + '.npy'))
        query_ids = np.load(os.path.join(data_dir, QUERY_IDS + '.npy'))
        features = np.load(os.path.join(data_dir, FEATURES + '.npy'))
        return relevance_labels, query_ids, features
    else:
        return None, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--n_hidden', type=int, help='n hidden units')
    parser.add_argument('--n_layers', type=int, help='n layers')
    parser.add_argument('--lambdarank', action='store_true')
    parser.add_argument('--factorized', action='store_true')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--n_features', type=int)
    parser.add_argument('--bn_mode', type=int)
    parser.add_argument('--epoch', type=int)
    args = parser.parse_args()

    np_train_file_directory = os.path.join(args.data_dir, 'np_train_files')
    train_relevance_labels, train_query_ids, train_features  = load_files(np_train_file_directory)

    np_test_file_directory = os.path.join(args.data_dir, 'np_test_files')
    test_relevance_labels, test_query_ids, test_features  = load_files(np_test_file_directory)

    np_vali_file_directory = os.path.join(args.data_dir, 'np_vali_files')
    vali_relevance_labels, vali_query_ids, vali_features  = load_files(np_vali_file_directory)

    learning_rate = 1e-5 if args.lr is None else args.lr
    network_desc = 'unfactorized'
    if args.factorized:
      network_desc = 'factorized'
    elif args.lambdarank:
      network_desc = 'lambdarank'
    print('Training a %s network, learning rate %f, n_hidden %s, n_layers %s' % (network_desc, learning_rate, args.n_hidden, args.n_layers))

    trainer = RankNetTrainer(args.n_hidden, train_relevance_labels, train_query_ids, train_features, test_relevance_labels,
                             test_query_ids, test_features, vali_relevance_labels, vali_query_ids, vali_features, args.model_dir)
    trainer.train(learning_rate, args.n_layers,  args.lambdarank, args.factorized, args.n_features, args.epoch)
