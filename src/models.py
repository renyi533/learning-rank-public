import tensorflow as tf
#import ipdb
import math
import numpy as np
import time

N_FEATURES = 136

def L2_loss(L2):
    loss = 0
    if L2 <= 0:
        return loss

    for v in tf.trainable_variables():
        loss = loss + tf.reduce_sum(tf.multiply(v, v))
    return loss * L2


def default_ranknet(x, relevance_labels, learning_rate, n_hidden, n_layers, n_features, enable_bn, L2, opt):
    N_FEATURES = n_features
    n_out = 1
    sigma = 1
    n_data = tf.shape(x)[0]

    def build_vars():
        variables = [tf.Variable(tf.random_normal([N_FEATURES, n_hidden], stddev=0.01)),
            tf.Variable(tf.zeros([n_hidden]))]
        if n_layers > 1:
            for i in range(n_layers-1):
                variables.append(tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=0.01)))
                variables.append(tf.Variable(tf.zeros([n_hidden])))
        variables.append(tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01)))
        variables.append(tf.Variable(0, dtype=tf.float32))
        print('Building an default ranknet neural network. learning_rate:%g, n_hidden:%d, n_layers:%d, n_features:%d, enable_bn:%s, L2:%g'
                % (learning_rate, n_hidden, n_layers, n_features, str(enable_bn), L2) )
        print(variables)
        return variables

    def score(x, *params):
        z = tf.matmul(x, params[0]) + params[1]
        if enable_bn:
            z = tf.contrib.layers.batch_norm(z, decay=0.999, center=True, scale=True)
        if n_layers > 1:
            for i in range(0,n_layers-1):
                z = tf.matmul(tf.nn.relu(z), params[2*(i+1)]) + params[2*(i+1)+1]
                if enable_bn:
                    z = tf.contrib.layers.batch_norm(z, decay=0.999, center=True, scale=True)
        return tf.matmul(tf.nn.relu(z), params[-2]) + params[-1]

    params = build_vars()
    predicted_scores = score(x, *params)
    #S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    S_ij = tf.sign(relevance_labels - tf.transpose(relevance_labels))
    real_scores = (1 / 2) * (1 + S_ij)
    pairwise_predicted_scores = predicted_scores - tf.transpose(predicted_scores)

    cost = tf.reduce_mean(
        (tf.ones([n_data, n_data]) - tf.diag(tf.ones([n_data]))) * tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pairwise_predicted_scores, labels=real_scores))

    cost = cost + L2_loss(L2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = opt.minimize(cost)

    def get_score(sess, feed_dict):
        return sess.run(predicted_scores, feed_dict=feed_dict)

    def run_optimizer(sess, feed_dict):
        sess.run(optimizer, feed_dict=feed_dict)

    return cost, run_optimizer, get_score

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

def default_lambdarank(x, relevance_labels, sorted_relevance_labels, index_range, learning_rate, n_hidden, n_layers, n_features, enable_bn, L2, ndcg_top, opt):
    N_FEATURES = n_features
    n_out = 1
    sigma = 1
    n_data = tf.shape(x)[0]

    def build_vars():
        variables = [tf.Variable(tf.random_normal([N_FEATURES, n_hidden], stddev=0.01)),
            tf.Variable(tf.zeros([n_hidden]))]
        if n_layers > 1:
            for i in range(n_layers-1):
                variables.append(tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=0.01)))
                variables.append(tf.Variable(tf.zeros([n_hidden])))
        variables.append(tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01)))
        variables.append(tf.Variable(0, dtype=tf.float32))
        print('Building an default lambdaRank neural network. learning_rate:%g, n_hidden:%d, n_layers:%d, n_features:%d, enable_bn:%s, L2:%g, trim_threshold:%d'
                % (learning_rate, n_hidden, n_layers, n_features, str(enable_bn), L2, ndcg_top) )
        print(variables)
        return variables

    def score(x, *params):
        z = tf.matmul(x, params[0]) + params[1]
        if enable_bn:
            z = tf.contrib.layers.batch_norm(z, decay=0.999, center=True, scale=True)
        if n_layers > 1:
            for i in range(0,n_layers-1):
                z = tf.matmul(tf.nn.relu(z), params[2*(i+1)]) + params[2*(i+1)+1]
                if enable_bn:
                    z = tf.contrib.layers.batch_norm(z, decay=0.999, center=True, scale=True)
        return tf.matmul(tf.nn.relu(z), params[-2]) + params[-1]

    params = build_vars()
    predicted_scores = score(x, *params)
    #S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    S_ij = tf.sign(relevance_labels - tf.transpose(relevance_labels))
    real_scores = (1 / 2) * (1 + S_ij)
    pairwise_predicted_scores = predicted_scores - tf.transpose(predicted_scores)

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
    if ndcg_top>0:
        swapped_ndcg = square_mask_tail_area(swapped_ndcg, ndcg_top)
    cost = tf.reduce_mean(
        (swapped_ndcg) * tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pairwise_predicted_scores, labels=real_scores))
    cost = cost + L2_loss(L2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = opt.minimize(cost)

    def get_score(sess, feed_dict):
        return sess.run(predicted_scores, feed_dict=feed_dict)

    def run_optimizer(sess, feed_dict):
        sess.run(optimizer, feed_dict=feed_dict)

    return cost, run_optimizer, get_score

def rnn_lambdarank(x, relevance_labels, sorted_relevance_labels, index_range, learning_rate, n_hidden, n_layers, n_features, enable_bn, step_cnt, L2, ndcg_top, lambdarank, opt):
    N_FEATURES = n_features
    n_out = 1
    sigma = 1
    n_data = tf.shape(x)[0]

    if lambdarank:
        name = 'lambdarank'
    else:
        name = 'ranknet'

    def build_vars():

        print('Building an rnn %s neural network. learning_rate:%g, n_hidden:%d, n_layers:%d, n_features:%d, enable_bn:%s, L2:%g, step_cnt:%d, trim_threshold:%d'
                % (name, learning_rate, n_hidden, n_layers, n_features, str(enable_bn), L2, step_cnt, ndcg_top) )
        return None

    def score(x):
      with tf.variable_scope("rnn"):
        seed_time = int(time.time())
        init_func = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed_time, dtype=tf.float32)

        features = tf.reshape(x, [-1, step_cnt, n_features])

        b = tf.get_variable("bias", [1], initializer=init_func)
        W = tf.get_variable("weights", [n_hidden, 1], initializer=init_func)

        rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=1.0, layer_norm=enable_bn)  for i in range(n_layers)], state_is_tuple=True)
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
        optimizer = opt.minimize(cost)

    def get_score(sess, feed_dict):
        return sess.run(predicted_scores, feed_dict=feed_dict)

    def run_optimizer(sess, feed_dict):
        sess.run(optimizer, feed_dict=feed_dict)

    return cost, run_optimizer, get_score

def ranknet(x, relevance_labels, learning_rate, n_hidden, build_vars_fn, score_with_batchnorm_update_fn, score_fn):
    n_out = 1
    sigma = 1
    n_data = tf.shape(x)[0]

    print('USING SIGMA = %f' % sigma)
    params = build_vars_fn()
    predicted_scores, bn_params = score_with_batchnorm_update_fn(x, params)
    #S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    S_ij = tf.sign(relevance_labels - tf.transpose(relevance_labels))
    real_scores = (1/2)*(1+S_ij)
    pairwise_predicted_scores = predicted_scores - tf.transpose(predicted_scores)
    lambdas = sigma*(1/2)*(1-S_ij) - sigma*tf.divide(1, (1 + tf.exp(sigma*pairwise_predicted_scores)))

    non_updating_predicted_scores = score_fn(x, bn_params, params)
    non_updating_S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    non_updating_real_scores = (1/2)*(1+non_updating_S_ij)
    non_updating_pairwise_predicted_scores = non_updating_predicted_scores - tf.transpose(non_updating_predicted_scores)
    non_updating_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=non_updating_pairwise_predicted_scores, labels=non_updating_real_scores))

    def get_derivative(W_k):
        dsi_dWk = tf.map_fn(lambda x_i: tf.squeeze(tf.gradients(score_fn(tf.expand_dims(x_i, 0), bn_params, params), [W_k])[0]), x)
        dsi_dWk_minus_dsj_dWk = tf.expand_dims(dsi_dWk, 1) - tf.expand_dims(dsi_dWk, 0)
        desired_lambdas_shape = tf.concat([tf.shape(lambdas), tf.ones([tf.rank(dsi_dWk_minus_dsj_dWk) - tf.rank(lambdas)], dtype=tf.int32)], axis=0)
        return tf.reduce_mean(tf.reshape(lambdas, desired_lambdas_shape)*dsi_dWk_minus_dsj_dWk, axis=[0,1])

    flat_params = [Wk for pk in params for Wk in pk]
    grads = [get_derivative(Wk) for Wk in flat_params]
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        adam_op = adam.apply_gradients([(tf.reshape(grad, tf.shape(param)), param) for grad, param in zip(grads, flat_params)])

    def optimizer(sess, feed_dict):
        sess.run(adam_op, feed_dict=feed_dict)

    def get_score(sess, feed_dict):
        return sess.run(non_updating_predicted_scores, feed_dict=feed_dict)

    return non_updating_cost, optimizer, get_score


def deep_factorized_ranknet(x, relevance_labels, learning_rate, n_hidden, n_layers, n_features, enable_bn):

    W = 0
    B = 1
    SCALE = 2
    BETA = 3
    MEAN = 0
    VAR = 1
    epsilon = 1e-3
    N_FEATURES = n_features
    def build_vars():
        if enable_bn:
            variables = [(tf.Variable(tf.random_normal([N_FEATURES, n_hidden], stddev=math.sqrt(2 / (N_FEATURES)))), # W_1
                tf.Variable(tf.zeros([n_hidden])), # b_1
                tf.Variable(tf.random_normal([n_hidden], stddev=1e-2) + 1),   # batchnorm, scale_1
                tf.Variable(tf.random_normal([n_hidden], stddev=1e-2))      # batchnorm, beta_1
                )]
        else:
            variables = [(tf.Variable(tf.random_normal([N_FEATURES, n_hidden], stddev=math.sqrt(2 / (N_FEATURES)))), # W_1
                tf.Variable(tf.zeros([n_hidden])) # b_1
                )]

        if n_layers > 1:
            for i in range(n_layers-1):
                if enable_bn:
                    variables.append((tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=math.sqrt(2 / (n_hidden)))), # W_i
                                 tf.Variable(tf.zeros([n_hidden])),  # b_i
                                 tf.Variable(tf.random_normal([n_hidden], stddev=1e-2) + 1),   # batchnorm, scale_i
                                 tf.Variable(tf.random_normal([n_hidden], stddev=1e-2)))) # batchnorm, beta_i
                else:
                    variables.append((tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=math.sqrt(2 / (n_hidden)))), # W_i
                                 tf.Variable(tf.zeros([n_hidden]))))  # b_i
        variables.append((tf.Variable(tf.random_normal([n_hidden, 1], stddev=math.sqrt(2 / (n_hidden)))),
                         tf.Variable(0, dtype=tf.float32)))
        print('Building an FACTORIZED neural network with the following layer parameters [W_1, b_1, scale_1, beta_1, ..., W_n, b_n]')
        print(variables)
        return variables

    def score(x, bn_params, params):
        z = tf.matmul(x, params[0][W]) + params[0][B]
        for i in range(0,n_layers):
            z_bn = z
            if enable_bn:
                # Normalize by mean and variance
                z1_hat = (z - bn_params[i][MEAN]) / tf.sqrt(bn_params[i][VAR] + epsilon)
                # Multiply by scale and add beta offset, learned params
                z_bn = params[i][SCALE]*z1_hat + params[i][BETA]
            # Apply non-linearity to batch normed z
            z = tf.matmul(tf.nn.relu(z_bn), params[i+1][W]) + params[i+1][B]
        return z

    def score_with_batchnorm_update(x, params):
        alpha = 1
        bn_params = list()
        z = tf.matmul(x, params[0][W]) + params[0][B]
        for i in range(0,n_layers):
            z_bn = z
            if enable_bn:
                # Update mean and variance
                batch_mean, batch_var = tf.nn.moments(z,[0])
                bn_params.append((batch_mean, batch_var))
                # Normalize by mean and variance
                z1_hat = (z - bn_params[i][MEAN]) / tf.sqrt(bn_params[i][VAR] + epsilon)
                # Multiply by scale and add beta offset, learned params
                z_bn = params[i][SCALE]*z1_hat + params[i][BETA]
            # Apply non-linearity to batch normed z
            z = tf.matmul(tf.nn.relu(z_bn), params[i+1][W]) + params[i+1][B]
        return z, bn_params

    return ranknet(x, relevance_labels, learning_rate, n_hidden, build_vars, score_with_batchnorm_update, score)


def lambdarank_deep(x, relevance_labels, sorted_relevance_labels, index_range, learning_rate, n_hidden, n_layers, n_features, enable_bn):
    W = 0
    B = 1
    SCALE = 2
    BETA = 3
    MEAN = 0
    VAR = 1
    epsilon = 1e-3
    N_FEATURES = n_features
    def build_vars():
        if enable_bn:
            variables = [(tf.Variable(tf.random_normal([N_FEATURES, n_hidden], stddev=math.sqrt(2 / (N_FEATURES)))), # W_1
                tf.Variable(tf.zeros([n_hidden])), # b_1
                tf.Variable(tf.random_normal([n_hidden], stddev=1e-2) + 1),   # batchnorm, scale_1
                tf.Variable(tf.random_normal([n_hidden], stddev=1e-2))      # batchnorm, beta_1
                )]
        else:
            variables = [(tf.Variable(tf.random_normal([N_FEATURES, n_hidden], stddev=math.sqrt(2 / (N_FEATURES)))), # W_1
                tf.Variable(tf.zeros([n_hidden])) # b_1
                )]

        if n_layers > 1:
            for i in range(n_layers-1):
                if enable_bn:
                    variables.append((tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=math.sqrt(2 / (n_hidden)))), # W_i
                                 tf.Variable(tf.zeros([n_hidden])),  # b_i
                                 tf.Variable(tf.random_normal([n_hidden], stddev=1e-2) + 1),   # batchnorm, scale_i
                                 tf.Variable(tf.random_normal([n_hidden], stddev=1e-2)))) # batchnorm, beta_i
                else:
                    variables.append((tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=math.sqrt(2 / (n_hidden)))), # W_i
                                 tf.Variable(tf.zeros([n_hidden]))))  # b_i

        variables.append((tf.Variable(tf.random_normal([n_hidden, 1], stddev=math.sqrt(2 / (n_hidden)))),
                         tf.Variable(0, dtype=tf.float32)))
        print('Building a LAMBDARANK neural network with the following layer parameters [W_1, b_1, scale_1, beta_1, ..., W_n, b_n]')
        print(variables)
        return variables

    def score(x, bn_params, params):
        z = tf.matmul(x, params[0][W]) + params[0][B]
        for i in range(0,n_layers):
            z_bn = z
            if enable_bn:
                # Normalize by mean and variance
                z1_hat = (z - bn_params[i][MEAN]) / tf.sqrt(bn_params[i][VAR] + epsilon)
                # Multiply by scale and add beta offset, learned params
                z_bn = params[i][SCALE]*z1_hat + params[i][BETA]
            # Apply non-linearity to batch normed z
            z = tf.matmul(tf.nn.relu(z_bn), params[i+1][W]) + params[i+1][B]
        return z

    def score_with_batchnorm_update(x, params):
        alpha = 1
        bn_params = list()
        z = tf.matmul(x, params[0][W]) + params[0][B]
        for i in range(0,n_layers):
            z_bn = z
            if enable_bn:
                # Update mean and variance
                batch_mean, batch_var = tf.nn.moments(z,[0])
                bn_params.append((batch_mean, batch_var))
                # Normalize by mean and variance
                z1_hat = (z - bn_params[i][MEAN]) / tf.sqrt(bn_params[i][VAR] + epsilon)
                # Multiply by scale and add beta offset, learned params
                z_bn = params[i][SCALE]*z1_hat + params[i][BETA]
            # Apply non-linearity to batch normed z
            z = tf.matmul(tf.nn.relu(z_bn), params[i+1][W]) + params[i+1][B]
        return z, bn_params

    return lambdarank(x, relevance_labels, sorted_relevance_labels, index_range, learning_rate, n_hidden, build_vars, score_with_batchnorm_update, score)

def lambdarank(x, relevance_labels, sorted_relevance_labels, index_range, learning_rate, n_hidden, build_vars_fn, score_with_batchnorm_update_fn, score_fn):
    n_out = 1
    sigma = 1
    n_data = tf.shape(x)[0]

    params = build_vars_fn()
    predicted_scores, bn_params = score_with_batchnorm_update_fn(x, params)
    #S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    S_ij = tf.sign(relevance_labels - tf.transpose(relevance_labels))
    real_scores = (1/2)*(1+S_ij)
    pairwise_predicted_scores = predicted_scores - tf.transpose(predicted_scores)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pairwise_predicted_scores, labels=real_scores))
    # # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf - lambdarank alternative
    # lambdas = -sigma*tf.divide(1, (1 + tf.exp(sigma*pairwise_predicted_scores)))
    lambdas = sigma*(1/2)*(1-S_ij) - sigma*tf.divide(1, (1 + tf.exp(sigma*pairwise_predicted_scores)))

    non_updating_predicted_scores = score_fn(x, bn_params, params)
    non_updating_S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    non_updating_real_scores = (1/2)*(1+non_updating_S_ij)
    non_updating_pairwise_predicted_scores = non_updating_predicted_scores - tf.transpose(non_updating_predicted_scores)
    non_updating_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=non_updating_pairwise_predicted_scores, labels=non_updating_real_scores))

    # LAMBDA RANK
    # Calculate the DCG, the IDCG, and then the difference between the DCG and every possible swapped DCG (an NxN matrix)
    log_2 = tf.log(tf.constant(2.0, dtype=tf.float32))
    cg_discount = tf.log(index_range+2)/log_2
    dcg = tf.reduce_sum( (relevance_labels) / cg_discount )
    idcg = tf.reduce_sum( (sorted_relevance_labels) / cg_discount )
    ndcg = dcg / idcg
    # remove the gain from label i then add the gain from label j
    stale_ij = tf.tile(((relevance_labels) / cg_discount), [1,n_data])
    new_ij = ((relevance_labels) / tf.transpose(cg_discount))
    stale_ji = tf.tile(tf.transpose((relevance_labels) / cg_discount), [n_data,1])
    new_ji = (tf.transpose(relevance_labels) / cg_discount)
    # if we swap i and j, we want to remove the stale CG term for i, add the new CG term for i,
    # remove the stale CG term for j, and then add the new CG term for j
    new_ndcg = (dcg - stale_ij + new_ij - stale_ji + new_ji) / idcg
    swapped_ndcg = tf.abs(ndcg - new_ndcg)
    lambdas_ndcg = lambdas*swapped_ndcg
    # # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf - lambdarank alternative
    # lambdas = tf.divide(1, (1 + tf.exp(pairwise_predicted_scores)))
    # lambdas_ndcg = (1/idcg) * lambdas * (2**relevance_labels - tf.transpose(2**relevance_labels)) * ((1 / tf.log(2 + index_range)) - tf.transpose(1 / tf.log(2 + index_range)))

    def get_derivative(W_k):
        dsi_dWk = tf.map_fn(lambda x_i: tf.squeeze(tf.gradients(score_fn(tf.expand_dims(x_i, 0), bn_params, params), [W_k])[0]), x)
        dsi_dWk_minus_dsj_dWk = tf.expand_dims(dsi_dWk, 1) - tf.expand_dims(dsi_dWk, 0)
        desired_lambdas_shape = tf.concat([tf.shape(lambdas_ndcg), tf.ones([tf.rank(dsi_dWk_minus_dsj_dWk) - tf.rank(lambdas_ndcg)], dtype=tf.int32)], axis=0)
        return tf.reduce_mean(tf.reshape(lambdas_ndcg, desired_lambdas_shape)*dsi_dWk_minus_dsj_dWk, axis=[0,1])

    flat_params = [Wk for pk in params for Wk in pk]
    grads = [get_derivative(Wk) for Wk in flat_params]
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        adam_op = adam.apply_gradients([(tf.reshape(grad, tf.shape(param)), param) for grad, param in zip(grads, flat_params)])

    def optimizer(sess, feed_dict):
        # a = [lambdas, idcg, relevance_labels, index_range, lambdas_ndcg]
        # ipdb.set_trace()
        sess.run(adam_op, feed_dict=feed_dict)

    def get_score(sess, feed_dict):
        return sess.run(predicted_scores, feed_dict=feed_dict)

    return cost, optimizer, get_score
