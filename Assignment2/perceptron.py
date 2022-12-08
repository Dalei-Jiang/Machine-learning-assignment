# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    shape = train_set.shape
    sample_amount = shape[0]
    weight = np.zeros(shape[1])
    bias = 0
    for j in range(0,max_iter):
        for i in range(sample_amount):
            train_sample = train_set[i]
            multi = bias * 1.0 + np.dot(weight, train_sample)
            if (multi <= 0 and train_labels[i] == 1):
                weight += train_sample*learning_rate
                bias += 1*learning_rate
            elif (multi > 0 and train_labels[i] == 0):
                weight -= train_sample*learning_rate
                bias -= 1*learning_rate
            else:
                continue
        
    W = weight
    b = bias
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # print(train_labels)
    weight, bias = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    # print(weight.size, type(bias))
    shape = dev_set.shape
    dev_label = [0]*shape[0]
    # weight_ex = np.append(weight, bias)
    # print(weight, bias)
    for i in range(shape[0]):
        dev_item = dev_set[i]
        # dev_item_ex = np.append(dev_item, 1.0)
        # print(dev_item_ex, weight_ex)
        multi = bias + np.vdot(dev_item.T, weight)
        if (multi <= 0):
            dev_label[i] = 0
        elif (multi > 0):
            dev_label[i] = 1
            
    return dev_label