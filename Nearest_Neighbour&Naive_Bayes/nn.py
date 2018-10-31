#!/usr/bin/env python

from scipy.io import loadmat
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random

def nn(X,Y,test):
    X_square = []
    for i in range(X.shape[0]):
        # print (i)
        X_square.append(np.dot(X[i], X[i]))
    X_square = np.tile(np.array(X_square), (test.shape[0], 1))
    #print("X_square constructed")
    test_X_prod = np.matmul(test, X.transpose())

    norm_equivalent = X_square - 2 * test_X_prod
    res_idx = np.argmin(norm_equivalent, axis=1)

    preds = np.array(list(map(lambda idx: Y[idx], res_idx)))
    return preds

if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    num_trials = 10
    for n in [ 1000, 2000, 4000, 8000 ]:
        test_err = np.zeros(num_trials)
        for trial in range(num_trials):
            sel = random.sample(range(len(ocr['data'].astype('float'))),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            test_err[trial] = np.mean(preds != ocr['testlabels'])
        print("%d\t%g\t%g" % (n,np.mean(test_err),np.std(test_err)))

