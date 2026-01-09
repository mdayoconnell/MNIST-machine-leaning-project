# Created by Micah
# Date: 1/9/26
# Time: 7:45 AM
# Project: NumpyNetwork
# File: 2-Layer-MLP.py

import numpy as np


def make_params(X, n_hidden, n_output):
    W1 = np.random.randn(n_hidden, X.shape[0]).astype(np.float32)
    b1 = np.random.randn(n_hidden, 1).astype(np.float32) / 100
    W2 = np.random.randn(n_output, n_hidden).astype(np.float32)
    b2 = np.random.randn(n_output, 1).astype(np.float32) / 100

    return W1, b1, W2, b2


def relu(x):  # Using ReLU here, if changing make sure to change relu_back_pass()
    return np.maximum(0, x)  # ReLU function


def relu_back_pass(dA, z):
    return dA * (z > 0)


def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    cache = (X, Z1, A1, Z2, A2)
    return A2, cache


def create_one_hot(Y):
    num_classes = len(np.unique(Y))
    m = Y.size
    one_hot = np.zeros((num_classes, m), dtype=bool)
    one_hot[Y, np.arange(m)] = 1
    return one_hot


def back_propagation(Y, W2, cache):
    X, Z1, A1, Z2, A2 = cache
    m = Y.size
    one_hot = create_one_hot(Y)
    dZ2 = A2 - one_hot
    dW2 = (1 / m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = relu_back_pass((W2.T @ dZ2), Z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


def calculate_accuracy(predictions, Yhat):
    return np.sum(predictions == Yhat) / Yhat.shape[0]


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def gradient_descent(X_train, Y_train, X_cref, Y_cref,
                     iterations=5000, alpha=0.05, k=100, n_hidden=10):


    #'''
    # alpha -> learning rate (initialized at 0.05)
    # k -> number of steps between accuracy checks
    #'''

    W1, b1, W2, b2 = make_params(X_train, n_hidden, 10)
    history = []

    A2_init, _ = forward_propagation(X_cref, W1, b1, W2, b2)
    predictions_init = get_predictions(A2_init)
    accuracy_init = calculate_accuracy(predictions_init, Y_cref)
    print("initial accuracy:", accuracy_init)
    history.append((-1, accuracy_init))

    for i in range(iterations):
        if i % k == 0:
            A2_cref, cache = forward_propagation(X_cref, W1, b1, W2, b2)
            predictions = get_predictions(A2_cref)
            accuracy = calculate_accuracy(predictions, Y_cref)
            history.append((i, accuracy))
            print("iteration:", i)
            print("Accuracy: ", accuracy)

        A2, cache = forward_propagation(X_train, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(Y_train, W2, cache)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

    return W1, b1, W2, b2, history
