from __future__ import print_function
import numpy as np
import time
import gzip

import multiprocessing
from functools import partial
from contextlib import contextmanager

image_size = 28
num_train = 50000
num_test = 10000

# read in training data
# images
f = gzip.open('train-images-idx3-ubyte.gz','r')
f.read(16)
buf = f.read(image_size * image_size * num_train)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_train = data.reshape(num_train, -1)
# labels
f = gzip.open('train-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(num_train)
y_train = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

# read in testing data
# images
f = gzip.open('t10k-images-idx3-ubyte.gz','r')
f.read(16)
buf = f.read(image_size * image_size * num_test)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
X_test = data.reshape(num_test, -1)
# labels
f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
f.read(8)
buf = f.read(num_test)
y_test = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

# this is the worker
def single_prediction(i, X, y, x, K=1, p=2, DEBUG = False):
    '''
        Inputs:
            i: current index
            X: A matrix of size N x F, which stores all the training data
            y: A vector of size N, which stores all the training labels
            x: A matrix of size M x F, which is the data we want to predict
            K: Number of Nearest Neighbour
            p: The power for distance metric
        Output:
            res: Predicted class
    '''
    # dist is vector of size N 
    dist = np.power( np.sum( np.power( np.abs( X-x[i] ), p), axis=1 ), 1./p)
    # dist_mask is of size K
    dist_mask = dist.argsort()[:K]
    # labels is of size K
    labels = y[dist_mask]
    if DEBUG:
        print(dist, dist_mask, labels)
    return np.bincount(labels).argmax()

def kNNPredict(X, y, x, K=1, p=2, DEBUG = False):
    '''
        Inputs:
            X: A matrix of size N x F, which stores all the training data

        Output:
            res: A matrix of size M x N, whose ith row stores the distance vector for x[i] 
    '''
    M = len(x)
    if __name__ == '__main__':
        with poolcontext(processes=16) as pool:
            res = pool.map(partial(single_prediction, X=X, y=y, x=x, K=K, p=p, DEBUG = DEBUG), range(M))
    return res

def kNNTrain(X, y, X_val, y_val, K=1, p=2):
    '''
        kNNTrain will help finding the optimal K
        Inputs:
            X: Training features of size N x F
            y: Training labels of size N
            X_val: Validation features of size M x F
            y_val: Validation labels of size M
            K: The number of neighbours
        Output:
            res: Accurancy for validation set
    '''
    predictions = kNNPredict(X, y, X_val, K=K, p=p)
    return np.float(np.sum(predictions == y_val)) / len(y_val)


# 10-fold cross validation
permutation = np.random.permutation(num_train)
X_train, y_train = X_train[permutation], y_train[permutation]
Ks = [1,2,3,4,5,10,20,50,100,500]
Accuracies = []

for k in Ks:
    accuracy = []
    step = int(num_train/10)
    for fold in range(0, num_train, step):
        mask = np.array([False] * num_train)
        mask[fold:fold+step] = True
        X = X_train[~mask]
        y = y_train[~mask]
        X_val = X_train[mask]
        y_val = y_train[mask]

        start_time = time.time()
        score = kNNTrain(X, y, X_val, y_val, K=k)
        elapsed_time = time.time() - start_time
        print("k=%d, accuracy=%.2f%%, time=%.2f" % (k, score*100, elapsed_time))
        accuracy.append(score)
    Accuracies.append(accuracy)

Accuracies = np.array(Accuracies)
np.savetxt("results.csv", Accuracies, delimiter=",")