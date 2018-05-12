import tensorflow as tf
#import ipdb
import math
import numpy as np
import time

N_FEATURES = 136
#export CUDA_VISIBLE_DEVICES=0
def L2_loss(L2):
    loss = 0
    if L2 <= 0:
        return loss

    for v in tf.trainable_variables():
        loss = loss + tf.reduce_sum(tf.multiply(v, v))
    return loss * L2

def strim_tensor(x, count, axis=0):
    if count<=0:
        return x

    length = tf.shape(x)[axis]
    return tf.cond(length > count, lambda: tf.split(x, [count, length-count], axis=axis)[0], lambda: x)


def _square_mask_tail_area(square, count):
    length = tf.shape(square)[0]
    top, bottom = tf.split(square, [count, length-count], axis=0)

    left, right = tf.split(bottom, [count, length-count], axis=1)

    bottom = tf.concat([left, tf.zeros([length-count, length-count])], axis=1)
    return tf.concat([top, bottom], axis=0)

def square_mask_tail_area(square, count):
    if count<=0:
        return square

    length = tf.shape(square)[0]

    return tf.cond(length <= count, lambda: square, lambda: _square_mask_tail_area(square, count))

def default_lambdarank(x, relevance_labels, sorted_relevance_labels, index_range, learning_rate,
        n_hidden, n_layers, n_features, enable_bn, L2, ndcg_top, lambdarank, opt, global_step, keep_prob, keep_prob_input):
    N_FEATURES = n_features
    n_out = 1
    sigma = 1
    n_data = tf.shape(x)[0]

    def build_vars():
        variables = [tf.Variable(tf.random_normal([N_FEATURES, n_hidden], stddev=0.001)),
            tf.Variable(tf.zeros([n_hidden]))]
        if n_layers > 1:
            for i in range(n_layers-1):
                variables.append(tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=0.001)))
                variables.append(tf.Variable(tf.zeros([n_hidden])))
        variables.append(tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.001)))
        variables.append(tf.Variable(0, dtype=tf.float32))
        print('Building an default lambdaRank neural network. learning_rate:%g, n_hidden:%d, n_layers:%d, n_features:%d, enable_bn:%s, L2:%g, trim_threshold:%d, use_lambda:%s'
                % (learning_rate, n_hidden, n_layers, n_features, str(enable_bn), L2, ndcg_top, str(lambdarank)) )
        print(variables)
        return variables

    def score(x, *params):
        x = tf.nn.dropout(x, keep_prob_input)
        z = tf.matmul(x, params[0]) + params[1]
        if enable_bn:
            z = tf.contrib.layers.batch_norm(z, decay=0.999, center=True, scale=True)
        if n_layers > 1:
            for i in range(0,n_layers-1):
                z = tf.matmul(tf.nn.dropout(tf.nn.relu(z), keep_prob), params[2*(i+1)]) + params[2*(i+1)+1]
                if enable_bn:
                    z = tf.contrib.layers.batch_norm(z, decay=0.999, center=True, scale=True)
        return tf.matmul(tf.nn.dropout(tf.nn.relu(z), keep_prob), params[-2]) + params[-1]

    params = build_vars()
    predicted_scores = score(x, *params)
    #S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    S_ij = tf.sign(relevance_labels - tf.transpose(relevance_labels))
    real_scores = (1 / 2) * (1 + S_ij)
    pairwise_predicted_scores = predicted_scores - tf.transpose(predicted_scores)

    if lambdarank:
        log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))
        cg_discount = tf.log(index_range+2)/log_2
        dcg = tf.reduce_sum( strim_tensor( (relevance_labels) / cg_discount , ndcg_top, axis=0) )
        idcg = tf.reduce_sum( strim_tensor(  ( (sorted_relevance_labels) / cg_discount ), ndcg_top, axis=0 ) )
        abs_idcg = tf.abs(idcg) + 1e-8
        ndcg = tf.cond(idcg > 0, lambda: dcg / abs_idcg, lambda: 0.0)
        # remove the gain from label i then add the gain from label j
        stale_ij = tf.tile(((relevance_labels) / cg_discount), [1,n_data])
        new_ij = ((relevance_labels) / tf.transpose(cg_discount))
        stale_ji = tf.tile(tf.transpose((relevance_labels) / cg_discount), [n_data,1])
        new_ji = (tf.transpose(relevance_labels) / cg_discount)
        # if we swap i and j, we want to remove the stale CG term for i, add the new CG term for i,
        # remove the stale CG term for j, and then add the new CG term for j
        new_ndcg = (dcg - stale_ij + new_ij - stale_ji + new_ji) / abs_idcg
        swapped_ndcg = tf.cond(idcg > 0, lambda: tf.abs(ndcg - new_ndcg), lambda: new_ndcg-new_ndcg)
    else:
        swapped_ndcg = (tf.ones([n_data, n_data]) - tf.diag(tf.ones([n_data])))

    if ndcg_top>0:
        swapped_ndcg = square_mask_tail_area(swapped_ndcg, ndcg_top)
    cost = tf.reduce_mean(
        (swapped_ndcg) * tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pairwise_predicted_scores, labels=real_scores))
    cost = cost + L2_loss(L2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = opt.minimize(cost, global_step=global_step)

    def get_score(sess, feed_dict):
        return sess.run(predicted_scores, feed_dict=feed_dict)

    def run_optimizer(sess, feed_dict):
        sess.run(optimizer, feed_dict=feed_dict)

    return cost, run_optimizer, get_score

def rnn_lambdarank(x, relevance_labels, sorted_relevance_labels, index_range, learning_rate, n_hidden, n_layers, n_features,
        enable_bn, step_cnt, L2, ndcg_top, lambdarank, rnn_type, opt, global_step, keep_prob, keep_prob_input):
    N_FEATURES = n_features
    n_out = 1
    sigma = 1
    n_data = tf.shape(x)[0]

    if lambdarank:
        name = 'lambdarank'
    else:
        name = 'ranknet'

    def build_vars():

        print('Building an rnn %s neural network. learning_rate:%g, n_hidden:%d, n_layers:%d, n_features:%d, enable_bn:%s, L2:%g, step_cnt:%d, trim_threshold:%d, rnn_type:%d'
                % (name, learning_rate, n_hidden, n_layers, n_features, str(enable_bn), L2, step_cnt, ndcg_top, rnn_type) )
        return None

    def score(x):
      with tf.variable_scope("rnn", initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001)):
        #seed_time = int(time.time())
        #init_func = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed_time, dtype=tf.float32)
        x = tf.nn.dropout(x, keep_prob_input)

        features = tf.reshape(x, [-1, step_cnt, n_features])
        b = tf.get_variable("bias", [1])
        W = tf.get_variable("weights", [n_hidden, 1])

        rnn_cell = None
        if rnn_type == 0:
            rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=keep_prob, layer_norm=enable_bn, activation=tf.nn.relu)  for i in range(n_layers)], state_is_tuple=True)
        else:
            rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(n_hidden, activation=tf.nn.relu)  for i in range(n_layers)], state_is_tuple=True)

        #data = tf.to_float(features)
        data = tf.transpose(features, [1,0,2])
        data = tf.reshape(data, [-1, n_features])
        data = tf.split(data, step_cnt, 0)
        outputs, status = tf.nn.static_rnn(rnn_cell, data, dtype=tf.float32)

        ouput = tf.add(tf.matmul(outputs[-1], W), b)

        return ouput

    params = build_vars()
    predicted_scores = score(x)
    #S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    S_ij = tf.sign(relevance_labels - tf.transpose(relevance_labels))
    real_scores = (1 / 2) * (1 + S_ij)
    pairwise_predicted_scores = predicted_scores - tf.transpose(predicted_scores)

    if lambdarank:
        log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))
        cg_discount = tf.log(index_range+2)/log_2
        dcg = tf.reduce_sum( strim_tensor( (relevance_labels) / cg_discount , ndcg_top, axis=0) )
        idcg = tf.reduce_sum( strim_tensor(  ( (sorted_relevance_labels) / cg_discount ), ndcg_top, axis=0 ) )
        abs_idcg = tf.abs(idcg) + 1e-8
        ndcg = tf.cond(idcg > 0, lambda: dcg / abs_idcg, lambda: 0.0)
        # remove the gain from label i then add the gain from label j
        stale_ij = tf.tile(((relevance_labels) / cg_discount), [1,n_data])
        new_ij = ((relevance_labels) / tf.transpose(cg_discount))
        stale_ji = tf.tile(tf.transpose((relevance_labels) / cg_discount), [n_data,1])
        new_ji = (tf.transpose(relevance_labels) / cg_discount)
        # if we swap i and j, we want to remove the stale CG term for i, add the new CG term for i,
        # remove the stale CG term for j, and then add the new CG term for j
        new_ndcg = (dcg - stale_ij + new_ij - stale_ji + new_ji) / abs_idcg
        swapped_ndcg = tf.cond(idcg > 0, lambda: tf.abs(ndcg - new_ndcg), lambda: new_ndcg-new_ndcg)
    else:
        swapped_ndcg = (tf.ones([n_data, n_data]) - tf.diag(tf.ones([n_data])))

    if ndcg_top>0:
        swapped_ndcg = square_mask_tail_area(swapped_ndcg, ndcg_top)

    cost = tf.reduce_mean(
        (swapped_ndcg) * tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pairwise_predicted_scores, labels=real_scores))
    cost = cost + L2_loss(L2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = opt.minimize(cost, global_step=global_step)

    def get_score(sess, feed_dict):
        return sess.run(predicted_scores, feed_dict=feed_dict)

    def run_optimizer(sess, feed_dict):
        sess.run(optimizer, feed_dict=feed_dict)

    return cost, run_optimizer, get_score

