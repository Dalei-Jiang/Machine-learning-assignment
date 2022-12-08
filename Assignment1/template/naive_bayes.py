# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    # print('The size of len(train_set) is', type(dev_set))
    # print('The size of len(train_labels) is', len(dev_labels))
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:
    for i in range(len(y)):
        if y[i] == 1:
            for j in X[i]:
            # throughout all the entries in the train_set[i]
                if pos_vocab.__contains__(j):
                    pos_vocab[j] = pos_vocab[j] + 1
                else:
                    pos_vocab[j] = 1
        elif y[i] == 0:
            for k in X[i]:
            # throughout all the entries in the train_set[i]
                if neg_vocab.__contains__(k):
                    neg_vocab[k] = neg_vocab[k] + 1
                else:
                    neg_vocab[k] = 1
                    
    # print('The type is', type(pos_vocab))
#    raise RuntimeError("The running time for the create_word_maps_uni has en error!")
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: pairs of words
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:
    for i in range(len(y)):
        if y[i] == 1:
            for j in X[i]:
            # throughout all the entries in the train_set[i]
                if pos_vocab.__contains__(j):
                    pos_vocab[j] = pos_vocab[j] + 1
                else:
                    pos_vocab[j] = 1
        elif y[i] == 0:
            for k in X[i]:
            # throughout all the entries in the train_set[i]
                if neg_vocab.__contains__(k):
                    neg_vocab[k] = neg_vocab[k] + 1
                else:
                    neg_vocab[k] = 1
                    
    for i in range(len(y)):
        if y[i] == 1:
            for j in range(len(X[i])-1):
            # throughout all the entries in the train_set[i]
                tar_key = str(X[i][j]) + ' ' + str(X[i][j+1])
                if pos_vocab.__contains__(tar_key):
                    pos_vocab[tar_key] = pos_vocab[tar_key] + 1
                else:
                    pos_vocab[tar_key] = 1
        elif y[i] == 0:
            for k in range(len(X[i])-1):
            # throughout all the entries in the train_set[i]
                tar_key = str(X[i][k]) + ' ' + str(X[i][k+1])
                if neg_vocab.__contains__(tar_key):
                    neg_vocab[tar_key] = neg_vocab[tar_key] + 1
                else:
                    neg_vocab[tar_key] = 1
                    
    # print('The type is', type(pos_vocab))
        
#    raise RuntimeError("The running time for the create_word_maps_bi has en error!")
    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)
    
    dev_labels = []
    neg_prior = 1.00 - pos_prior
    # pro_words = 1.00
    p_voca, n_voca = create_word_maps_uni(train_set, train_labels, max_size=None)
    Number_0 = 0
    Number_1 = 0
    for i in p_voca.values():
        Number_1 += i
    for j in n_voca.values():
        Number_0 += j    
    Abs_0 = len(n_voca)
    Abs_1 = len(p_voca)

    for mail in dev_set:
        LOG_Pro_ham = math.log(neg_prior)
        LOG_Pro_spam = math.log(pos_prior)
        for word in mail:
            P_word_0 = (n_voca.get(word, 0) + laplace) / (Number_0 + laplace*(1 + Abs_0))
            P_word_1 = (p_voca.get(word, 0) + laplace) / (Number_1 + laplace*(1 + Abs_1))
            LOG_P_word_0 = math.log(P_word_0)
            LOG_P_word_1 = math.log(P_word_1)
            LOG_Pro_ham += LOG_P_word_0
            LOG_Pro_spam += LOG_P_word_1
        # Pro_ham = math.e**(LOG_Pro_ham)
        # Pro_spam = math.e**(LOG_Pro_spam)
        if (LOG_Pro_ham >= LOG_Pro_spam):
            dev_labels.append(0)
        else:
            dev_labels.append(1)
    
    # raise RuntimeError("Unfortunately, the running time has encountered an error in NaiveBayes!")
    # print(dev_labels)
    return dev_labels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.9,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    max_vocab_size = None
    
    dev_labels = []
    pos_voca_uni, neg_voca_uni = create_word_maps_uni(train_set, train_labels, max_size=None)
    pos_voca_bin, neg_voca_bin = create_word_maps_bi(train_set, train_labels, max_size=None)
    unigram_lambda = 1.00 - bigram_lambda
    neg_prior = 1.00 - pos_prior
    Num_uni_0 = 0
    Num_uni_1 = 0
    Num_bin_0 = 0
    Num_bin_1 = 0
    for i in pos_voca_uni.values():
        Num_uni_1 += i
    for j in neg_voca_uni.values():
        Num_uni_0 += j
    for k in pos_voca_bin.values():
        Num_bin_1 += k
    for l in neg_voca_bin.values():
        Num_bin_0 += l
    
    Abs_uni_0 = len(neg_voca_uni)
    Abs_uni_1 = len(pos_voca_uni)
    Abs_bin_0 = len(neg_voca_bin)
    Abs_bin_1 = len(pos_voca_bin)
    for mail in dev_set:
        LOG_Pro_ham_uni = math.log(neg_prior)
        LOG_Pro_spam_uni = math.log(pos_prior)       
        for word in mail:
            P_word_0 = (neg_voca_uni.get(word, 0) + unigram_laplace) / (Num_uni_0 + unigram_laplace*(1 + Abs_uni_0))
            LOG_P_word_0 = math.log(P_word_0)
            LOG_Pro_ham_uni += LOG_P_word_0
            
            P_word_1 = (pos_voca_uni.get(word, 0) + unigram_laplace) / (Num_uni_1 + unigram_laplace*(1 + Abs_uni_1))
            LOG_P_word_1 = math.log(P_word_1)
            LOG_Pro_spam_uni += LOG_P_word_1
            
        LOG_Pro_ham_bin = math.log(neg_prior)
        LOG_Pro_spam_bin = math.log(pos_prior)
        for index in range(len(mail)-1):
            bigram = str(mail[index]) + ' ' + str(mail[index+1])
            P_bigram_0 = (neg_voca_bin.get(bigram, 0) + bigram_laplace) / (Num_bin_0 + bigram_laplace*(1 + Abs_bin_0))
            LOG_P_bigram_0 = math.log(P_bigram_0)
            LOG_Pro_ham_bin += LOG_P_bigram_0
            
            P_bigram_1 = (pos_voca_bin.get(bigram, 0) + bigram_laplace) / (Num_bin_1 + bigram_laplace*(1 + Abs_bin_1))
            LOG_P_bigram_1 = math.log(P_bigram_1)
            LOG_Pro_spam_bin += LOG_P_bigram_1
        
        LOG_Pro_ham = unigram_lambda*LOG_Pro_ham_uni + bigram_lambda*LOG_Pro_ham_bin
        LOG_Pro_spam = unigram_lambda*LOG_Pro_spam_uni + bigram_lambda*LOG_Pro_spam_bin
        if (LOG_Pro_ham >= LOG_Pro_spam):
            dev_labels.append(0)
        else:
            dev_labels.append(1)     
            
#    raise RuntimeError("Unfortunately, the running time has encountered an error in BigramBayes!")

    return dev_labels
