# Created by Micah
# Date: 1/9/26
# Time: 8:03 AM
# Project: NumpyNetwork
# File: mnist_csv.py

import pandas as pd
import numpy as np

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