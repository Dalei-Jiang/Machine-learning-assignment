# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
from math import log

def maxtag(dic):
    maxnum = -1
    for tag in dic.keys():
        if dic[tag] > maxnum:
            maxnum = dic[tag]
            tagmax = tag
    return tagmax

def adddict(a, dic):
    if a not in dic.keys():
        dic[a] = 1
    else:
        dic[a] += 1

def addlist(a, lis):
    if a not in lis:
        lis.append(a)

def dictsingle(dic, a, A):
    if a not in dic.keys():
        return A
    else:
        return dic[a]

def dicbin(dic, a, b, A):
    if a not in dic.keys():
        return A
    elif b not in dic[a].keys():
        return A
    else:
        return dic[a][b]

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    # Training process
    K = 0.0001
    A = 0.0001    
    T_sen = len(train)
    tagpair_dict = {} 
    tagword_dict = {}
    firsttag_dict = {}
    Tag_dict = {}
    Word_set = []
    Tag_set = []
    # training process
    counter = 0
    for sentence in train:
        counter += 1
        previous_pair = ("Alan","Turing")
        wordpair1 = sentence[0]
        tag1 = wordpair1[1]
        adddict(tag1, firsttag_dict)            
        for wordpair in sentence:
            word = wordpair[0]
            tag = wordpair[1]
            addlist(word, Word_set)
            addlist(tag, Tag_set)
            adddict(tag, Tag_dict)
            
            if previous_pair != ("Alan","Turing"):
                pre_tag = previous_pair[1]
                if pre_tag not in tagpair_dict.keys():
                    tagpair_dict[pre_tag] = {tag:1,}
                else:
                    adddict(tag, tagpair_dict[pre_tag])
            
            if tag not in tagword_dict.keys():
                tagword_dict[tag] = {word:1}
            else:
                adddict(word, tagword_dict[tag])
            previous_pair = wordpair
    
    # Data process
    N = len(Tag_set) # the number of distinct tags
    V = len(Word_set) # the number of distinct words
    # print(Tag_set)
    Aij_dict = {}
    Bjk_dict = {}
    Pij_dict = {}
    for tag in firsttag_dict.keys():
        prob_pi = (firsttag_dict[tag]+K)/(T_sen+K*N)
        Pij_dict[tag] = log(prob_pi)
    
    # print(tagpair_dict)
    for i in tagpair_dict.keys():
        if i not in Aij_dict.keys():
            Aij_dict[i] = {}
        j_dic = tagpair_dict[i]
        for j in j_dic:
            j_count = tagpair_dict[i][j]
            aij = (j_count+K)/(Tag_dict[i]+K*N)
            Aij_dict[i][j] = log(aij)
    
    for j in tagword_dict.keys():
        if j not in Bjk_dict.keys():
            Bjk_dict[j] = {}
        k_dic = tagword_dict[j]
        for k in k_dic:
            k_count = tagword_dict[j][k]
            bjk = (k_count + K)/(Tag_dict[j]+K*(V+1))
            Bjk_dict[j][k] = log(bjk)   
    
    
    counter2 = 0
    # Developing process
    final_article = []
    for sentence in test:
        counter2 += 1
        new_sent = []
        Backtrace_dict = {}
        counter = 0
        pre_count = 0
        for k in sentence:
            Viterbi_dict = {}
            current_word = k
            if counter == 0:
                for j in Tag_set:
                    pi_j = dictsingle(Pij_dict, j, log(A/(T_sen+A*N)))
                    bjk = dicbin(Bjk_dict, j, k, log(A/(Tag_dict[j]+A*(V+1))))
                    v = pi_j+bjk
                    Viterbi_dict[j] = v
                    Backtrace_dict[(k,j,counter)] = ("Alan","Turing")
                ex_word = k
                VXY_dict = Viterbi_dict.copy()
            else:
                for j in Tag_set:
                    bjk = dicbin(Bjk_dict, j, k, log(A/(Tag_dict[j]+A*(V+1))))
                    max_pair = (ex_word,Tag_set[0], pre_count)
                    max_value = VXY_dict[Tag_set[0]]+aij+bjk
                    for i in Tag_set:
                        aij = dicbin(Aij_dict, i, j, log(A/(Tag_dict[j]+A*(V+1))))
                        eij = aij+bjk
                        vi = VXY_dict[i]
                        vj = eij+vi
                        if vj > max_value:
                            max_pair = (ex_word, i, pre_count)
                            max_value = vj
                    Backtrace_dict[(current_word,j,counter)] = max_pair
                    Viterbi_dict[j] = max_value
                VXY_dict = Viterbi_dict.copy()
                ex_word = k
            pre_count = counter
            counter += 1
        
        last_dict = VXY_dict
        final_maxtag = list(last_dict.keys())[0]
        final_maxvalue = last_dict[final_maxtag]
        for j in last_dict.keys():
            prob = last_dict[j]
            if prob > final_maxvalue:
                final_maxvalue = prob
                final_maxtag = j
        target = (ex_word,j, pre_count)
        while (target!=("Alan","Turing")):
            pair = list(target)[0:2]
            new_sent.insert(0,pair)
            target = Backtrace_dict[target]
        # print(new_sent)
        final_article.append(new_sent)               
    return final_article