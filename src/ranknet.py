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
import math


class RankNetTrainer:
    def __init__(self, n_hidden, train_relevance_labels, train_query_ids, train_features, test_relevance_labels,
                 test_query_ids, test_features, vali_relevance_labels, vali_query_ids, vali_features, model_dir, ndcg_top,
                 beta1, beta2, epsilon):
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


        self.unique_ids = self.train_unique_query_ids
        np.random.shuffle(self.unique_ids)

        self.ndcg_top = ndcg_top
        self.models_directory = model_dir

        self.train_sample_dict = { }
        self.vali_sample_dict = { }

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
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

    def train(self, learning_rate, n_layers, lambdarank, factorized, n_features, epoch, enable_bn, L2):
        x = tf.placeholder("float", [None, n_features])
        relevance_scores = tf.placeholder("float", [None, 1])
        sorted_relevance_scores = tf.placeholder("float", [None, 1])
        index_range = tf.placeholder("float", [None, 1], name='index_range')
        lr = tf.placeholder("float", [])
        query_indices = tf.placeholder("float", [None])
        self.learning_rate = learning_rate
        self.start_time = time.time()
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)
        print('Adam parameters: learning_rate:%f, beta1:%f, beta2:%f, epsilon:%g' %(self.learning_rate, self.beta1, self.beta2, self.epsilon))
        if lambdarank:
            self.filename = 'nn_lambdarank_%slayers_%shidden_lr%s' % (n_layers, self.n_hidden, ('%.0E' % self.learning_rate).replace('-', '_'))
            cost, optimizer, score = models.default_lambdarank(x, relevance_scores, sorted_relevance_scores, index_range,
                                                               learning_rate, self.n_hidden, n_layers, n_features, enable_bn, L2, opt)
        elif not factorized:
            self.filename = 'nn_unfactorized_ranknet_%slayers_%shidden_lr%s' % (n_layers, self.n_hidden, ('%.0E' % self.learning_rate).replace('-', '_'))
            cost, optimizer, score = models.default_ranknet(x, relevance_scores, learning_rate, self.n_hidden, n_layers, n_features, enable_bn, L2, opt)
        elif factorized:
            self.filename = 'nn_factorized_ranknet_%slayers_%shidden_lr%s' % (n_layers, self.n_hidden, ('%.0E' % self.learning_rate).replace('-', '_'))
            cost, optimizer, score = models.deep_factorized_ranknet(x, relevance_scores, self.learning_rate, self.n_hidden, n_layers, n_features, enable_bn)
        else:
            raise('Need to specify if this model should be unfactorized, factorized, or use lambdarank!')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            ckpt = tf.train.get_checkpoint_state(self.models_directory)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path+". Will load saved model")
                saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
                epoch = 0
            else:
                print('no valid saved model found')

            if self.train_features is None:
                print('no training data provided. quit training')
                epoch = 0

            if epoch > 0:
                for c_id in self.unique_ids:
                    indices = np.where(self.train_query_ids == c_id)[0]
                    self.train_sample_dict[c_id] = indices

                if self.vali_features is not None:
                    for c_id in self.vali_unique_query_ids:
                        indices = np.where(self.vali_query_ids == c_id)[0]
                        self.vali_sample_dict[c_id] = indices

            print("Trainable variables are:")
            for v in tf.trainable_variables():
              print("parameter:", v.name, "device:", v.device, "shape:", v.get_shape())
            print("Global variables are:")
            for v in tf.global_variables():
              print("parameter:", v.name, "device:", v.device, "shape:", v.get_shape())

            c_iter = 0
            while c_iter<epoch:

                np.random.shuffle(self.unique_ids)
                #indices = np.random.randint(1, len(self.train_features), batch_size)
                for index in range(len(self.unique_ids)):
                    c_id = self.unique_ids[index]
                    orig_indices = self.train_sample_dict[c_id]
                    curr_train_features = self.train_features[orig_indices]
                    curr_train_labels = self.train_relevance_labels[orig_indices]

                    if lambdarank:
                        optimizer(sess, {
                            x: np.array(curr_train_features, ndmin=2),
                            relevance_scores: np.array(curr_train_labels, ndmin=2).T,
                            lr: self.learning_rate,
                            query_indices: indices,
                            index_range: np.array([float(i) for i in range(0,len(curr_train_labels))], ndmin=2).T,
                            sorted_relevance_scores: np.sort(np.array(curr_train_labels, ndmin=2)).T[::-1]
                        })
                    else:
                        optimizer(sess, {
                            x: np.array(curr_train_features, ndmin=2),
                            relevance_scores: np.array(curr_train_labels, ndmin=2).T,
                            lr: self.learning_rate,
                            query_indices: indices
                        })
                if c_iter % 1 == 0:
                    self.check_progress(sess, saver, cost, score, x, relevance_scores, c_iter,index_range, sorted_relevance_scores, True)
                c_iter += 1
            if self.test_features is not None:
                test_avg_cost, test_avg_err, test_avg_ndcg, test_avg_full_ndcg = self.check_scores(cost,
                  self.test_features,
                  self.test_query_ids,
                  self.test_relevance_labels,
                  relevance_scores, score, sess,
                  self.test_unique_query_ids, x, index_range, sorted_relevance_scores)
                print('Test Cost: {:10f} NDCG: {:9f} ({:9f}) ERR: {:9f}  {:9f} s'.format(
                        test_avg_cost, test_avg_ndcg, test_avg_full_ndcg, test_avg_err, time.time() - self.start_time))
                predictions = self.compute_predictions(self.test_features, self.test_relevance_labels, relevance_scores, score, sess, x)
                filename = os.path.join(self.models_directory, 'pred.csv')
                with open(filename, 'w') as f:
                    for elem in predictions:
                        f.write(str(elem[0])+'\n')


    def check_progress(self, sess, saver, cost, score, x, relevance_scores, c_iter, index_range, sorted_relevance_scores, save_data=True):
        train_avg_cost, train_avg_err, train_avg_ndcg, train_avg_full_ndcg = self.check_scores(cost,
            self.train_features,
            self.train_query_ids,
            self.train_relevance_labels,
            relevance_scores, score, sess,
            self.train_unique_query_ids, x, index_range, sorted_relevance_scores, self.train_sample_dict)

        if self.vali_features is not None:
            vali_avg_cost, vali_avg_err, vali_avg_ndcg, vali_avg_full_ndcg = self.check_scores(cost,
              self.vali_features,
              self.vali_query_ids,
              self.vali_relevance_labels,
              relevance_scores, score, sess,
              self.vali_unique_query_ids, x, index_range, sorted_relevance_scores, self.vali_sample_dict)
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
                     unique_query_ids, x, index_range, sorted_relevance_scores, sample_dict=None):
        costs = list()
        ndcg_scores = list()
        full_ndcg_scores = list()
        err_scores = list()
        assert len(unique_query_ids) > 0
        for c_id in unique_query_ids:
            if sample_dict is not None:
                query_indices = sample_dict[c_id]
            else:
                query_indices = np.where(query_ids == c_id)[0]
            c_cost = sess.run(cost, feed_dict={
                x: np.array(features[query_indices], ndmin=2),
                relevance_scores: np.array(relevance_labels[query_indices], ndmin=2).T,
                index_range: np.array([float(i) for i in range(0,len(query_indices))], ndmin=2).T,
                sorted_relevance_scores: np.sort(np.array(relevance_labels[query_indices], ndmin=2)).T[::-1]
                })
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
            sorted_query_indices = scored_pred_query['query_int']
            ndcg_scores.append(ndcg(relevance_labels[sorted_query_indices], top_count=self.ndcg_top))
            full_ndcg_scores.append(ndcg(relevance_labels[sorted_query_indices], top_count=None))
            if sample_dict is not None:
                sample_dict[c_id] = sorted_query_indices
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

def read_libsvm_data(filename):
    start_time = time.time()
    label_list = list()
    features = list()
    current_row = 0
    print('start loading data: '+filename)
    with open(filename, 'r') as f:
        for line in f:
            q2 = line.split(" ")
            label_list.append(q2[0])
            del q2[0]
            d = ':'.join(map(str, q2))
            e = d.split(":")
            features.append(e[1::2])
            if current_row % 10000 == 0:
                print('row %d - %f seconds' % (current_row, time.time() - start_time))
                #print('label:'+str(label_list[current_row]))
                #print('features:'+str(features[current_row]))
            current_row += 1

    print('Done loading data: %s - %f seconds' % (filename, time.time() - start_time))

    label_list = np.asarray(label_list, dtype=float)
    features = np.asarray(features, dtype=float)
    query_ids = np.asarray(features[:, 0], dtype=int)
    features = features[:, 1:]
    return label_list, query_ids, features

def read_data(data_dir):
    if data_dir is not None and os.path.exists(data_dir):
        return read_libsvm_data(data_dir)
    else:
        return None, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--L2', type=float, help='weight decay', default=0.00)
    parser.add_argument('--beta1', type=float, help='weight decay', default=0.9)
    parser.add_argument('--beta2', type=float, help='weight decay', default=0.999)
    parser.add_argument('--epsilon', type=float, help='weight decay', default=1e-8)
    parser.add_argument('--n_hidden', type=int, help='n hidden units', default=50)
    parser.add_argument('--n_layers', type=int, help='n layers', default=1)
    parser.add_argument('--lambdarank', action='store_true', default=False)
    parser.add_argument('--factorized', action='store_true', default=False)
    parser.add_argument('--enable_bn', action='store_true', default=False)
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--validation_data', type=str)
    parser.add_argument('--n_features', type=int, default=59)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--ndcg_top', type=int, default=1500)
    args = parser.parse_args()


    train_relevance_labels, train_query_ids, train_features = read_data(args.train_data)
    test_relevance_labels, test_query_ids, test_features = read_data(args.test_data)
    vali_relevance_labels, vali_query_ids, vali_features = read_data(args.validation_data)

    learning_rate = 1e-5 if args.lr is None else args.lr
    enable_bn = False if args.enable_bn is None else args.enable_bn
    network_desc = 'unfactorized'
    if args.factorized:
      network_desc = 'factorized'
    elif args.lambdarank:
      network_desc = 'lambdarank'
    print('Training a %s network, learning rate %f, n_hidden %s, n_layers %s' % (network_desc, learning_rate, args.n_hidden, args.n_layers))

    trainer = RankNetTrainer(args.n_hidden, train_relevance_labels, train_query_ids, train_features, test_relevance_labels,
                             test_query_ids, test_features, vali_relevance_labels, vali_query_ids, vali_features, args.model_dir, args.ndcg_top,
                             args.beta1, args.beta2, args.epsilon)
    trainer.train(learning_rate, args.n_layers,  args.lambdarank, args.factorized, args.n_features, args.epoch, enable_bn, args.L2)
