#!/usr/bin/env python3
"""
    Optimization project
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ Shuffles data points in two matrices the same way
        X_train is ndarray with shape (m, 784) containing traning data
            m number of data points
            784 is number of features
        Y_train is ndarray with shape (m, 10) containing training labels
            m number of data points
            10 number of classes
        X_valid (m, 784) contains validation data
        Y_valid is one-hot (m, 10), validation labels
        batch_size
        epochs is # of times training passes through whole dataset
        load_path/save_path where model stored
        Returns: path where saved
        allow for smaller final batch
        shuffle training data before each epoch
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)
        m = X_train.shape[0]
        batches = m / batch_size
        if batches % 1 != 0:
            batches = int(batches) + 1
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        for i in range(epochs + 1):

            train_cost = loss.eval({x: X_train,
                                    y: Y_train})
            train_accuracy = accuracy.eval({x: X_train,
                                            y: Y_train})
            valid_cost = loss.eval({x: X_valid,
                                    y: Y_valid})
            valid_accuracy = accuracy.eval({x: X_valid,
                                            y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))
            if i == epochs:
                ''' Done training, last epoch metrics printed '''
                break
            shuf_x, shuf_y = shuffle_data(X_train, Y_train)
            for j in range(batches):
                start = batch_size * j
                end = batch_size * (j + 1)
                '''if end > m:
                    end = -1'''
                sess.run(train_op, feed_dict={x: shuf_x[start:end],
                                              y: shuf_y[start:end]})
                if (j + 1) % 100 == 0 and j != 0:
                    step_cost = sess.run(loss, {x: shuf_x[start:end],
                                                y: shuf_y[start:end]})
                    step_accuracy = sess.run(accuracy, {x: shuf_x[start:end],
                                                        y: shuf_y[start:end]})
                    print("\tStep {}:".format(j + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))
        return saver.save(sess, save_path)
