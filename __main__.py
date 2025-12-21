# Created by Micah
# Date: 12/14/25
# Time: 10:59 AM
# Project: NumpyNetwork
# File: __main__.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# '''
# Initializing Data from Kaggle
# '''

def create_mnist_csv(csv_path: str, sanity_check=False):
    '''
    Kaggle digit recognizer takes the czv, returns an array with label, pixel1, pixel2...
    returns:
    Y - label array, dims 1 x m
    X - pixel array, dims 784 x m

    Sanity check should give dimensions and values:
    X: (784 x m)
    Y:(m,)

    labels min/max = 0, 9
    pixel range min/max = 0.0, 1.0

    '''
    df = pd.read_csv(csv_path)          # Kaggle makes this as a csv, we make it a numpy array
    data = df.to_numpy()

    Y = data[:, 0].astype(np.int64)         # This data is the label 0-9
    X = (data[:, 1:].astype(np.float32) / 255.0).T  # (784, m)      # This data is the pixel value as float, 0-1

    if sanity_check:
        print("X:", X.shape, "Y:", Y.shape)
        print("labels min/max:", Y.min(), Y.max())
        print("pixel range:", X.min(), X.max())

    return X, Y

def split_data(X, Y, cref_fraction=0.1):
    '''
    Shuffles and splits our data into sets for training and cross-reference
    '''
    rng = np.random.default_rng(42)
    m = X.shape[1]
    index = rng.permutation(m)
    cref_size = int(m * cref_fraction)

    cref_index = index[:cref_size]
    train_index = index[cref_size:]

    X_train = X[:, train_index]
    Y_train = Y[train_index]
    X_cref = X[:, cref_index]
    Y_cref = Y[cref_index]

    return X_train, Y_train, X_cref, Y_cref

# '''
# Setting up Model, defining activation and softmax functions
# '''

def make_params(X, n_hidden, n_output):
    W1 = np.random.randn(n_hidden,X.shape[0]).astype(np.float32)
    b1 = np.random.randn(n_hidden,1).astype(np.float32) / 100
    W2 = np.random.randn(n_output, n_hidden).astype(np.float32)
    b2 = np.random.randn(n_output, 1).astype(np.float32) / 100

    return W1, b1, W2, b2

def relu(x):          # Using ReLU here, if changing make sure to change relu_back_pass()
    return np.maximum(0,x)  # ReLU function

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
    one_hot = np.zeros((num_classes,m), dtype=bool)
    one_hot[Y, np.arange(m)] = 1
    return one_hot

def back_propagation(Y, W2, cache):
    X, Z1, A1, Z2, A2 = cache
    m = Y.size
    one_hot = create_one_hot(Y)
    dZ2 = A2 - one_hot
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = relu_back_pass((W2.T @ dZ2), Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
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

# '''
#   This step makes predictions and changes the model accordingly, training, updating weights, calculating accuracy, repeating
# '''

def gradient_descent(X_train, Y_train, X_cref, Y_cref,
                     iterations=50000, alpha=0.05, k=100, n_hidden=10):

    W1, b1, W2, b2 = make_params(X_train,n_hidden,10)
    history = []

    # '''
    #   Initial Accuracy Test
    # '''

    A2_init,_ = forward_propagation(X_cref, W1, b1, W2, b2)
    predictions_init = get_predictions(A2_init)
    accuracy_init = calculate_accuracy(predictions_init, Y_cref)
    print("initial accuracy:", accuracy_init)
    history.append((-1,accuracy_init))

    # '''
    #   The training:
    #       Forward propagation gives us probabilities in output layer A2
    #       Back propagation compares to the answer, calculates how to adjust weights
    #       Update parameters accordingly
    # '''

    for i in range(iterations):

        if i % k == 0:
            A2_cref, cache = forward_propagation(X_cref, W1, b1, W2, b2)
            predictions = get_predictions(A2_cref)
            accuracy = calculate_accuracy(predictions, Y_cref)
            history.append((i,accuracy))
            print("iteration:", i)
            print("Accuracy: ", accuracy)

        A2, cache = forward_propagation(X_train, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(Y_train, W2, cache)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

    # '''
    #   We compute accuracy every k iterations.
    #   Run get_predictions() on our cross reference pixels (this returns an A2-style vector)
    #   calculate_accuracy() compares that to our cross reference labels
    # '''





    A2_cref, cache = forward_propagation(X_cref, W1, b1, W2, b2)

    return W1, b1, W2, b2, history


# '''
# Run
# '''

def load_plot(n_hidden, i, acc):
    training_comparison = dict([])

if __name__ == "__main__":

    all_histories = {}
    X, Y = create_mnist_csv("/Users/micah/kaggle/digit-recognizer/train.csv")

    # Take 90% of the data for training, 10% for cross referencing
    X_train, Y_train, X_cref, Y_cref = split_data(X,Y,cref_fraction=0.1)


    for hidden_neurons_size in range(8,9):
        print("training with hidden units: ", int(2**hidden_neurons_size))
        print("initial accuracy: ", calculate_accuracy(get_predictions(X_cref), Y_cref))

        W1, b1, W2, b2, history = gradient_descent(
            X_train, Y_train, X_cref, Y_cref,
            n_hidden=int(2**hidden_neurons_size),
        )

        all_histories[2**hidden_neurons_size] = history


    for h, hist in all_histories.items():
        iters = [t for t, _ in hist]
        accs = [a for _, a in hist]
        plt.plot(iters, accs, label=f'hidden neurons: {h}')
    plt.xlabel("Iteration")
    plt.ylabel("Cross-reference accuracy")
    plt.title("MNIST accuracy vs number of forward training propagations")
    plt.legend()
    plt.show()

    #print("final accuracy", calculate_accuracy(get_predictions(A2_cref),Y_cref))
