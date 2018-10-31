#!/usr/bin/env python



from __future__ import print_function
from scipy.io import loadmat
import numpy as np
import math

def estimate_naive_bayes_classifier(X,Y):
    params = {}
    training_size = X.shape[0]
    label_num = len(set(Y))
    d = X.shape[1]

    cond_prob = np.zeros([label_num,d])
    posterior = np.zeros(label_num)
    for i in range(label_num):
        class_size = np.sum(Y==i+1)
        posterior[i] = float(class_size)/training_size
        temp = np.sum(X[np.where(Y==i+1)[0]],axis = 0)
        cond_prob[i] = (1+temp)/float(2+class_size)

    #print (posterior)
    params['first_term'] = np.sum(np.log(1 - cond_prob), axis=1)+np.log(posterior)
    params['cond_prob_equi'] = np.log( cond_prob.transpose()/(1-cond_prob.transpose()) )
    params['cond_prob'] = cond_prob
    params['posterior'] = posterior
    return params



def predict(params,X):
    first_term = params['first_term']
    cond_prob = params['cond_prob_equi']
    n = X.shape[0]

    prob_matrix = np.dot(X.toarray(), cond_prob )
    #print ("mult finished")
    prob_matrix += np.tile(first_term,(n,1))
    res = np.argmax(prob_matrix,axis = 1)+1
    return res

def print_top_words(params,vocab):
    cond_prob = params['cond_prob_equi']
    alpha = cond_prob[:,1] - cond_prob[:,0]
    large_idx = alpha.argsort()[-20:][::-1]
    small_idx = alpha.argsort()[:20]
    vocab = np.array(vocab)
    print ('Positive words:',vocab[large_idx])
    print('negative words:', vocab[small_idx])
def load_data():
    return loadmat('news.mat')

def load_vocab():
    with open('news.vocab') as f:
        vocab = [ x.strip() for x in f.readlines() ]
    return vocab

if __name__ == '__main__':
    news = load_data()

    # 20-way classification problem

    data = news['data']
    labels = news['labels'][:,0]
    testdata = news['testdata']
    testlabels = news['testlabels'][:,0]

    params = estimate_naive_bayes_classifier(data,labels)
    pred = predict(params,data) # predictions on training data
    testpred = predict(params,testdata) # predictions on test data

    print('20 classes: training error rate: %g' % np.mean(pred != labels))
    print('20 classes: test error rate: %g' % np.mean(testpred != testlabels))

    # binary classification problem

    indices = (labels==1) | (labels==16) | (labels==20) | (labels==17) | (labels==18) | (labels==19)
    data2 = data[indices,:]
    labels2 = labels[indices]
    labels2[(labels2==1) | (labels2==16) | (labels2==20)] = 0
    labels2[(labels2==17) | (labels2==18) | (labels2==19)] = 1
    testindices = (testlabels==1) | (testlabels==16) | (testlabels==20) | (testlabels==17) | (testlabels==18) | (testlabels==19)
    testdata2 = testdata[testindices,:]
    testlabels2 = testlabels[testindices]
    testlabels2[(testlabels2==1) | (testlabels2==16) | (testlabels2==20)] = 0
    testlabels2[(testlabels2==17) | (testlabels2==18) | (testlabels2==19)] = 1

    params2 = estimate_naive_bayes_classifier(data2,labels2+1)
    pred2 = predict(params2,data2) # predictions on training data
    testpred2 = predict(params2,testdata2) # predictions on test data

    print('2 classes: training error rate: %g' % np.mean(pred2 != labels2+1))
    print('2 classes: test error rate: %g' % np.mean(testpred2 != testlabels2+1))

    vocab = load_vocab()
    print_top_words(params2,vocab)

