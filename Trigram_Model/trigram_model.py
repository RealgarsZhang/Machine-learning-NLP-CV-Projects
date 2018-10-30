import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""
"""

"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    if n == 1:
        res = [('START',)]
    else:
        res = []
    l = 0
    r = n-1
    sequence = ['START']*(n-1) + sequence + ['STOP'] # doing copy. a little inefficient
    while r<len(sequence):
        res.append(tuple(sequence[l:r+1]))
        l += 1
        r += 1
    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        ##Your code here
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            unigrams = get_ngrams(sentence,1)
            for t in unigrams:
                self.unigramcounts[t] += 1
            bigrams = get_ngrams(sentence,2)
            self.bigramcounts[('START','START')] += 1 # there are trigrams like START,START,the
            for t in bigrams:
                self.bigramcounts[t] += 1
            trigrams = get_ngrams(sentence,3)
            for t in trigrams:
                self.trigramcounts[t] += 1

        self.total_wc_with_STOP = sum( self.unigramcounts.values() )-self.unigramcounts[('START',)]

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[trigram[:2]]==0:
            return 0.0
        return float(self.trigramcounts[trigram])/self.bigramcounts[trigram[:2]]


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        return float(self.bigramcounts[bigram])/self.unigramcounts[bigram[:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """


        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return float(self.unigramcounts[unigram])/self.total_wc_with_STOP

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        res = lambda3*self.raw_trigram_probability(trigram) +\
              lambda2*self.raw_bigram_probability(trigram[1:])+\
              lambda1*self.raw_unigram_probability(trigram[2:])
        return res
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigram_collection = get_ngrams(sentence,3)
        res = 0
        for trigram in trigram_collection:
            res += math.log2( self.smoothed_trigram_probability(trigram) )

        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        word_cnt = 0
        sum_l = 0

        for sentence in corpus:
            word_cnt += len(sentence)
            sum_l += self.sentence_logprob(sentence)

        return 2**( -sum_l/word_cnt )


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        # WLOG, let 1 be high, 2 be low
        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1): # high labels
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            correct += int(pp1<pp2)
    
        for f in os.listdir(testdir2): # low labels
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total += 1
            correct += int(pp1 > pp2)
        
        return float(correct)/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    print ( "Testing Perplexity:", model.perplexity(corpus_reader("./hw1_data/brown_test.txt",lexicon = model.lexicon)) )
    print ("Training Perplexity:", model.perplexity(corpus_reader("./hw1_data/brown_train.txt",lexicon = model.lexicon)) )

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt',"train_low.txt", "test_high", "test_low")
    # print(acc)
    toefl_dir = "./hw1_data/ets_toefl_data/"
    acc = essay_scoring_experiment(toefl_dir+'train_high.txt', toefl_dir+"train_low.txt", toefl_dir+"test_high", toefl_dir+"test_low")
    print ("Accuracy: ",acc)

