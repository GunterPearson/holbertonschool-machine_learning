#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
one_hot = __import__('3-one_hot').one_hot
load_model = __import__('9-model').load_model
predict = __import__('13-predict').predict


if __name__ == '__main__':
    datasets = np.load('../data/MNIST.npz')
    X_valid_3D = datasets['X_test']
    X_test = datasets['X_test']
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_test = datasets['Y_test']

    network = load_model('network2.h5')
    Y_pred = predict(network, X_test)
    print(Y_pred)
    print(np.argmax(Y_pred, axis=1))
    print(Y_test)
    Z = np.argmax(Y_pred, axis=1)

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_valid_3D[i])
        plt.title("{} {}".format(Y_test[i], Z[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
