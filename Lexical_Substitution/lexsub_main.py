#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np
import string
from collections import defaultdict, Counter
import scipy

# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = set()
    l = wn.lemmas("_".join(lemma.split()),pos = pos)
    #print (l)
    for item in l:
        s = item.synset()
        for le in s.lemmas():
            string = le.name()
            string = " ".join(string.split('_'))
            if string != lemma:
                possible_synonyms.add(string.lower())

    return list(possible_synonyms)

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    freq = defaultdict(int)
    l = wn.lemmas("_".join(context.lemma.split()),context.pos)

    for lexeme in l:
        s = lexeme.synset()
        for le in s.lemmas():
            name = le.name()
            name = " ".join(name.split('_'))
            if name != context.lemma:
                #freq[name] += le.count()*lexeme.count()
                freq[name] += le.count()
                #freq[name] += lexeme.count()#the question is a little ambiguous
    max_name = ""
    max_freq = -1
    #print(freq)
    for k in freq:
        if freq[k]>max_freq:
            max_name = k
            max_freq = freq[k]


    return max_name # replace for part 2

def my_predictor1(context):
    freq = defaultdict(int)
    l = wn.lemmas("_".join(context.lemma.split()),context.pos)

    for lexeme in l:
        s = lexeme.synset()
        for le in s.lemmas():
            name = le.name()
            name = " ".join(name.split('_'))
            if name != context.lemma:
                freq[name] += le.count()*lexeme.count()
    max_name = ""
    max_freq = -1
    #print(freq)
    for k in freq:
        if freq[k]>max_freq:
            max_name = k
            max_freq = freq[k]


    return max_name



def get_eg_def(s):
    res = tokenize(s.definition())

    for eg in s.examples():
        res += tokenize(eg)

    for hyper_s in s.hypernyms():
        res += tokenize(hyper_s.definition())
        for eg in hyper_s.examples():
            res += tokenize(eg)
    return set(res)

def common_cnt(s_words,context_words,stop_words):# context words is a set
    cnt = 0
    for word in context_words:
        if word not in stop_words and word in s_words:
            cnt += 1

    return cnt



def wn_simple_lesk_predictor(context):
    l = wn.lemmas("_".join(context.lemma.split()), context.pos)
    stop_words = set(stopwords.words("english"))
    cur_max_overlap = 0
    overlap_dic = defaultdict(list)
    context_words = set(context.left_context + context.right_context)
    for lexeme in l:
        s = lexeme.synset()
        s_words = get_eg_def(s)
        #print (s_words)

        if context.lemma in context_words:
            context_words.remove(context.lemma)# how to handle the _
        cnt = common_cnt(s_words,context_words,stop_words)

        if cnt > cur_max_overlap:
            cur_max_overlap = cnt

        overlap_dic[cnt].append(lexeme)

    #return ""
    for cnt in sorted(overlap_dic.keys(),reverse = True):
        res_list = overlap_dic[cnt]
        res_list.sort(key = lambda lx:lx.count(),reverse = True)
        for lexeme in res_list:
            s = lexeme.synset()
            candidates = [l for l in s.lemmas()]
            candidates.sort(key = lambda l:l.count(),reverse = True)
            for c in candidates:
                if c.name() != "_".join(context.lemma.split()):
                    return " ".join(c.name().split("_"))

    #print(res_list)
    #print(cur_max_overlap)

def get_window(context,stop_words):
    window_size =5
    res = []
    prog = 0
    i = 0
    while prog<window_size:
        try:
            while context.right_context[i].lower() in stop_words:# or context.right_context[i] in string.punctuation:
                i+=1
            res.append(context.right_context[i].lower())
            i+=1
        except IndexError:
            break
        prog += 1

    prog = 0
    i = 1
    while prog < window_size:
        try:
            while context.left_context[-i].lower() in stop_words:# or context.left_context[-i] in string.punctuation:
                i += 1
            res.append(context.left_context[-i].lower())
            i += 1
        except IndexError:
            break
        prog += 1

    return res

def cos(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

''' 
class trial(object):
    def __init__(self, model):
        self.model = model

    def predict_nearest_with_context(self, context):
        stop_words = set(stopwords.words("english"))
        sentence = get_window(context,stop_words)
        #print (sentence)
        #print (context.left_context,context.right_context)
        candidates = get_candidates(context.lemma,context.pos)
        sentence_vec = np.array(self.model.wv[context.lemma])
        for word in sentence:
            if word in self.model.wv:
                sentence_vec += self.model.wv[word]
        #print(sentence_vec)
        #print(np.linalg.norm(sentence_vec.astype("float")))
        res = ""
        dis = -1
        for w in candidates:
            word = "_".join(w.split())
            try:
                cur_dis = 1-scipy.spatial.distance.cosine(sentence_vec,self.model.wv[word])
                #cur_dis = cos(sentence_vec, self.model.wv[word])

            except KeyError:
                #print (word)
                continue
            if cur_dis>dis:
                dis = cur_dis
                res = word
        return " ".join(res.split('_')) # replace for part 5

'''


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        candidates = get_candidates(context.lemma,context.pos)
        target = context.lemma
        res = ""
        sim = -10
        for w in candidates:
            word = "_".join(w.split())
            try:
                cur_sim = self.model.similarity(word,target)
            except KeyError:
                continue
            if cur_sim>sim:
                res = word
                sim = cur_sim

        return " ".join(res.split('_')) # replace for part 4

    def predict_nearest_with_context(self, context):
        stop_words = set(stopwords.words("english"))
        sentence = get_window(context,stop_words)
        #print (sentence)
        #print (context.left_context,context.right_context)
        candidates = get_candidates(context.lemma,context.pos)
        sentence_vec = self.model.wv[context.lemma].astype("float")
        for word in sentence:
            if word in self.model.wv:
                sentence_vec += self.model.wv[word].astype("float")
        res = ""
        dis = -1
        for w in candidates:
            word = "_".join(w.split())
            try:
                cur_dis = cos(sentence_vec,self.model.wv[word].astype("float"))
            except KeyError:
                #print (word)
                continue
            if cur_dis>dis:
                dis = cur_dis
                res = word
        return " ".join(res.split('_')) # replace for part 5

    """

    def my_predictor2(self, context):
        stop_words = set(stopwords.words("english"))
        sentence = get_window(context, stop_words)
        # print (sentence)
        # print (context.left_context,context.right_context)
        candidates = get_candidates(context.lemma, context.pos)
        sentence_vec = self.model.wv[context.lemma].copy()
        for word in sentence:
            if word in self.model.wv:
                sentence_vec += self.model.wv[word]
        sentence_vec /= (len(sentence)+1)
        res = ""
        dis = 9999999
        for w in candidates:
            word = "_".join(w.split())
            try:
                cur_dis = scipy.spatial.distance.euclidean(sentence_vec, self.model.wv[word])

            except KeyError:
                continue
            if cur_dis < dis:
                dis = cur_dis
                res = word
        return " ".join(res.split('_'))  # replace for part 5
    """


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = './GoogleNews-vectors-negative300.bin'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest_with_context(context)
        #prediction = predictor.predict_nearest(context)
        #prediction = predictor.my_predictor2(context)
        prediction = my_predictor1(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))

