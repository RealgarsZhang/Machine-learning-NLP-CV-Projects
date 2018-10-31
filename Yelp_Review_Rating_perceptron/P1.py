import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
from collections import Counter,defaultdict


def get_ngrams(text, n):
    sequence = text.split()
    l = 0
    r = n-1
    #sequence = ['START']*(n-1) + sequence + ['STOP'] # doing copy. a little inefficient
    res = []
    while r<len(sequence):
        res.append(tuple(sequence[l:r+1]))
        l += 1
        r += 1
    return res



def unigram(train_corpus,train_labels,test_corpus,test_labels):
    vectorizer = CountVectorizer()
    # append one at last
    train_features = vectorizer.fit_transform(train_corpus)
    train_size = train_features.shape[0]
    temp = sp.sparse.csr_matrix(np.ones((train_size,1)))
    train_features = sp.sparse.csr_matrix(sp.sparse.hstack([train_features,temp]))
    train_features = sp.sparse.csr_matrix(train_features, dtype="uint32") #modify the data type for idf!!!!

    test_features = vectorizer.transform(test_corpus)
    test_size = test_features.shape[0]
    temp = sp.sparse.csr_matrix(np.ones((test_size, 1)))
    test_features = sp.sparse.csr_matrix(sp.sparse.hstack([test_features,temp]))
    test_features = sp.sparse.csr_matrix(test_features, dtype="uint32")

    d = train_features.shape[1]
    n = train_features.shape[0]

    print("Unigram: Features extracted...")
    index = np.arange(n)
    np.random.shuffle(index)
    shuffled_train_features1 = train_features[index,:]
    shuffled_train_labels1 = train_labels[index]

    np.random.shuffle(index)
    shuffled_train_features2 = train_features[index, :]
    shuffled_train_labels2 = train_labels[index]

    # first pass
    w_cur = np.zeros(d,dtype = 'int')
    for i in range(n//5):
        #10*i,10*i+1,10*i+2,...,10*i+9

        s = 5*i
        print(s)
        vec = shuffled_train_features1[s,:]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label-pred)/2*vec


        s = 5 * i+1
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

        s = 5 * i+2
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

        s = 5 * i+3
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

        s = 5 * i+4
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

    for s in range(5*i+5,n):
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

    #second pass
    w_sum = w_cur.copy()
    for i in range(n//5):

        #10*i,10*i+1,10*i+2,...,10*i+9
        s = 5*i
        print(s)
        vec = shuffled_train_features2[s,:]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label-pred)/2*vec
        w_sum = w_sum + w_cur

        s = 5 * i+1
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

        s = 5 * i+2
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

        s = 5 * i+3
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

        s = 5 * i+4
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

    for s in range(5*i+5,n):
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

    print("training finished")
    w = w_sum/(n+1)
    train_preds = np.sign(train_features.dot(w)).reshape((train_size,))
    test_preds = np.sign(test_features.dot(w)).reshape((test_size,))
    #print(train_preds)
    err_tr = np.sum(train_preds!=np.array(train_labels))/train_size
    print (np.sum(train_preds!=np.array(train_labels)))
    print (train_size)
    err_te = np.sum(test_preds!=np.array(test_labels))/test_size

    print("Training error:" ,err_tr)
    print("Testing Error:",err_te)

    return [w,train_features,test_features,vectorizer]

def tfidf(train_corpus,train_labels,test_corpus,test_labels):
    vectorizer = TfidfVectorizer()
    # append one at last
    train_features = vectorizer.fit_transform(train_corpus)
    train_size = train_features.shape[0]
    temp = sp.sparse.csr_matrix(np.ones((train_size,1)))
    train_features = sp.sparse.csr_matrix(sp.sparse.hstack([train_features,temp]))
    train_features = sp.sparse.csr_matrix(train_features) #modify the data type for idf!!!!

    test_features = vectorizer.transform(test_corpus)
    test_size = test_features.shape[0]
    temp = sp.sparse.csr_matrix(np.ones((test_size, 1)))
    test_features = sp.sparse.csr_matrix(sp.sparse.hstack([test_features,temp]))
    test_features = sp.sparse.csr_matrix(test_features)

    d = train_features.shape[1]
    n = train_features.shape[0]

    print("Tfidf: Features extracted...")
    index = np.arange(n)
    np.random.shuffle(index)
    shuffled_train_features1 = train_features[index,:]
    shuffled_train_labels1 = train_labels[index]

    np.random.shuffle(index)
    shuffled_train_features2 = train_features[index, :]
    shuffled_train_labels2 = train_labels[index]

    # first pass
    w_cur = np.zeros(d)
    for i in range(n//5):
        #10*i,10*i+1,10*i+2,...,10*i+9

        s = 5*i
        print(s)
        vec = shuffled_train_features1[s,:]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label-pred)/2*vec

        s = 5 * i+1
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

        s = 5 * i+2
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

        s = 5 * i+3
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

        s = 5 * i+4
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

    for s in range(5*i+5,n):
        vec = shuffled_train_features1[s, :]
        label = shuffled_train_labels1[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec

    #second pass
    w_sum = w_cur.copy()
    for i in range(n//5):

        #10*i,10*i+1,10*i+2,...,10*i+9
        s = 5*i
        print(s)
        vec = shuffled_train_features2[s,:]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label-pred)/2*vec
        w_sum = w_sum + w_cur

        s = 5 * i+1
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

        s = 5 * i+2
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

        s = 5 * i+3
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

        s = 5 * i+4
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

    for s in range(5*i+5,n):
        vec = shuffled_train_features2[s, :]
        label = shuffled_train_labels2[s]
        pred = np.sign(vec.dot(w_cur)-0.001)
        w_cur = w_cur + (label - pred) / 2 * vec
        w_sum = w_sum + w_cur

    print("training finished")
    w = w_sum/(n+1)
    train_preds = np.sign(train_features.dot(w)).reshape((train_size,))
    test_preds = np.sign(test_features.dot(w)).reshape((test_size,))
    #print(train_preds)
    err_tr = np.sum(train_preds!=np.array(train_labels))/train_size
    print (np.sum(train_preds!=np.array(train_labels)))
    print (train_size)
    err_te = np.sum(test_preds!=np.array(test_labels))/test_size

    print("Training error:" ,err_tr)
    print("Testing Error:",err_te)

    return [w,train_features,test_features,vectorizer]


def dict_subtract(weight,l):
    for item in l:
        weight[item] -= 1

def dict_add(weight,l):
    for item in l:
        weight[item] += 1

def dict_sum(w1,w2):
    for key in w1:
        w1[key] += w2[key]


def ngram_perceptron(train_corpus,train_labels,test_corpus,test_labels,highest_gram):
    n = train_corpus.shape[0]

    index = np.arange(n)

    temp1 = train_corpus.copy()
    temp2 = train_labels.copy()
    shuffled_data1 = np.vstack((temp1, temp2))
    shuffled_data1 = shuffled_data1.transpose()
    np.random.shuffle(shuffled_data1)

    temp1 = train_corpus.copy()
    temp2 = train_labels.copy()
    shuffled_data2 = np.vstack((temp1, temp2))
    shuffled_data2 = shuffled_data2.transpose()
    np.random.shuffle(shuffled_data2)

    weight_dic = defaultdict(int)
    for i in range(n):
        print(i)
        text = shuffled_data1[i][0]
        label = shuffled_data1[i][1]
        bi_seq = []
        for k in range(1,highest_gram+1):
            bi_seq += get_ngrams(text, k)
        prod = 0
        for item in bi_seq:
            prod += weight_dic[item]
        prod += weight_dic[1]
        #print(prod, label)
        if prod * label <= 0:
            if label < 0:
                dict_subtract(weight_dic, bi_seq)
                weight_dic[1] -= 1
            else:
                dict_add(weight_dic, bi_seq)
                weight_dic[1] += 1

    weight_sum = weight_dic.copy()

    for key in weight_sum:
        weight_sum[key] *= (n + 1)

    # trial_weight_sum = weight_dic.copy()
    for i in range(n):
        print(i)
        text = shuffled_data2[i][0]
        label = shuffled_data2[i][1]
        bi_seq = []
        for k in range(1, highest_gram + 1):
            bi_seq += get_ngrams(text, k)
        prod = 0
        for item in bi_seq:
            prod += weight_dic[item]
        prod += weight_dic[1]

        if prod * label <= 0:
            if label < 0:
                dict_subtract(weight_dic, bi_seq)
                weight_dic[1] -= 1
                for item in bi_seq:
                    weight_sum[item] -= (n - i)
                weight_sum[1] -= (n - i)
            else:
                dict_add(weight_dic, bi_seq)
                weight_dic[1] += 1
                for item in bi_seq:
                    weight_sum[item] += (n - i)
                weight_sum[1] += (n - i)
                # dict_sum(trial_weight_sum,weight_dic)

    for key in weight_sum:
        weight_sum[key] /= (n + 1)
        # trial_weight_sum[key] /= (n+1)

    test_size = test_corpus.shape[0]
    wrong_cnt = 0
    for i in range(test_size):
        text = test_corpus[i]
        label = test_labels[i]
        bi_seq = []
        for k in range(1, highest_gram + 1):
            bi_seq += get_ngrams(text, k)
        prod = 0
        for item in bi_seq:
            prod += weight_sum[item]
        prod += weight_sum[1]

        if prod * label <= 0:
            wrong_cnt += 1

    print("Test error is :", wrong_cnt / test_size)

    wrong_cnt = 0
    for i in range(n):
        text = train_corpus[i]
        label = train_labels[i]
        bi_seq = []
        for k in range(1, highest_gram + 1):
            bi_seq += get_ngrams(text, k)
        prod = 0
        for item in bi_seq:
            prod += weight_sum[item]
        prod += weight_sum[1]

        if prod * label <= 0:
            wrong_cnt += 1

    print("Train error is :", wrong_cnt / n)

train_filename = "reviews_tr.csv"
test_filename = "reviews_te.csv"

##############################
#train_filename = "trial_train.csv"
#test_filename = "trial_test.csv"
##############################
traindata = pd.read_csv(train_filename)
train_corpus = traindata['text']
train_labels = np.array(traindata['rating']*2-1,dtype="int8")

testdata = pd.read_csv(test_filename)
test_corpus = testdata['text']
test_labels = np.array(testdata['rating']*2-1,dtype="int8")

print("Data read in...")

buffer = input("Press ENTER to run for unigram.")

reply_unigram = unigram(train_corpus,train_labels,test_corpus,test_labels)
vectorizer = reply_unigram[-1]
weight = reply_unigram[0]
weight = weight[:-1]
words = np.array(vectorizer.get_feature_names())

largest_10_idx = weight.argsort()[-10:]
smallest_10_idx = weight.argsort()[:10]

largest_10 = words[largest_10_idx]
smallest_10 = words[smallest_10_idx]

print("Most positive words:",sorted(largest_10))
print("Most negative words:",sorted(smallest_10))

wrong_idx = []
test_features = reply_unigram[2]
prog = 0
weight = reply_unigram[0][:-1]
weight = weight.reshape((1,len(weight)))
while len(wrong_idx)<2:
    pred = np.sign(test_features[prog].dot(reply_unigram[0]))
    label = test_labels[prog]
    if label!=pred:
        print("Correct label:",label,"Predicted label:",pred)
        print("Index:",prog)
        print("Misclassified text:",test_corpus[prog])
        vec = test_features[prog][:, :-1].multiply(weight)
        vec = np.array(vec.todense())
        vec = vec.reshape((vec.shape[1],))
        largest_10_idx = vec.argsort()[-10:]
        smallest_10_idx = vec.argsort()[:10]
        print("words with max values:", words[largest_10_idx])
        print("words with min values:", words[smallest_10_idx])
        wrong_idx.append(prog)
    prog += 1


buffer = input("Press ENTER to run for tfidf.")
reply_tfidf = tfidf(train_corpus,train_labels,test_corpus,test_labels)
buffer = input("Press ENTER to run for bigram.")
ngram_perceptron(train_corpus,train_labels,test_corpus,test_labels,2)
buffer = input("Press ENTER to run for trigram.")
ngram_perceptron(train_corpus,train_labels,test_corpus,test_labels,3)










