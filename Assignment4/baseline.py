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
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
def maxtag(dic):
    maxnum = -1
    for tag in dic.keys():
        if dic[tag] > maxnum:
            maxnum = dic[tag]
            tagmax = tag
    return tagmax

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # training process
    All_dict = {}
    Tag_dict = {}
    for sentence in train:
        for wordpair in sentence:
            word = wordpair[0]
            tag = wordpair[1]
            if word not in All_dict.keys():
                All_dict[word] = {tag:1}
            else:
                if tag not in All_dict[word].keys():
                    All_dict[word][tag] = 1
                else:
                    All_dict[word][tag] += 1
            if tag not in Tag_dict.keys():
                Tag_dict[tag] = 1
            else:
                Tag_dict[tag] += 1
    common_tag = maxtag(Tag_dict)
    
    # developing process
    final_article = []
    for sentence in test:
        new_sen = []
        for word in sentence:
            if word not in All_dict.keys():
                tagmax = common_tag
            else:
                tagmax = maxtag(All_dict[word])
            new_sen.append((word, tagmax))
        final_article.append(new_sen)
        
    return final_article