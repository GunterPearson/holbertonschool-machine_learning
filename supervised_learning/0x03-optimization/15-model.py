#!/usr/bin/env python3
""" together"""
import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """ shuffle data"""
    s = np.random.permutation(X.shape[0])
    return X[s], Y[s]


def create_placeholders(nx, classes):
    """Create placeholders"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes),  name="y")
    return x, y


def create_batch_norm_layer(prev, n, activation):
    """ create batch normilization layer"""
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=w)
    new = layer(prev)
    gamma = tf.Variable(tf.ones((1, n)), trainable=True)
    beta = tf.Variable(tf.zeros((1, n)), trainable=True)
    mean, variance = tf.nn.moments(new, axes=[0])
    r = tf.nn.batch_normalization(new, mean, variance,
                                  beta, gamma, 1e-8)
    return activation(r)


def create_layer(prev, n, activation):
    """ create layer"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    L = tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=W, name="layer")
    return L(prev)


def forward_prop(x, layer_sizes=[], activations=[]):
    """ forward prop"""
    L = x
    for i in range(len(layer_sizes)):
        if activations[i] is None:
            L = create_layer(L, layer_sizes[i],
                             activations[i])
        else:
            L = create_batch_norm_layer(L, layer_sizes[i],
                                        activations[i])
    return L


def calculate_accuracy(y, y_pred):
    """ calculate accuracy"""
    p_m = tf.argmax(y_pred, 1)
    y_m = tf.argmax(y, 1)
    e = tf.equal(y_m, p_m)
    return tf.reduce_mean(tf.cast(e, tf.float32))


def calculate_loss(y, y_pred):
    """ loss function"""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ update with adam tensorflow"""
    a = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return a.minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ learning rate decay"""
    a = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                    decay_rate, staircase=True)
    return a


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """ model using ADAM"""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    global_step = tf.Variable(0)
    decay = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, decay, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        size = X_train.shape[0] // batch_size
        if X_train.shape[0] % batch_size != 0:
            size += 1
        for i in range(epochs + 1):
            cost_t, acc_t = sess.run([loss, accuracy],
                                     feed_dict={x: X_train, y: Y_train})
            cost_v, acc_v = sess.run([loss, accuracy],
                                     feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
            if i < epochs:
                x_sh, y_sh = shuffle_data(X_train, Y_train)
                for j in range(size):
                    start = j * batch_size
                    end = start + batch_size
                    if end > x_sh.shape[0]:
                        end = x_sh.shape[0]
                    x_mini = x_sh[start:end]
                    y_mini = y_sh[start:end]
                    sess.run(train_op, feed_dict={x: x_mini, y: y_mini})
                    if (j + 1) % 100 == 0 and j > 0:
                        cost, acc = sess.run([loss, accuracy],
                                             feed_dict={x: x_mini, y: y_mini})
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(cost))
                        print("\t\tAccuracy: {}".format(acc))
            sess.run(tf.assign(global_step, global_step + 1))
        return saver.save(sess, save_path)
