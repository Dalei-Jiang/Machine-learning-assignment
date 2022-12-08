# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques for the fall 2021 semester
# Modified by Kaiwen Hong for the Spring 2022 semester

"""
This is the main entry point for MP2. You should only modify code
within this file and neuralnet.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class1 = 1
class2 = 3

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @param l(x,y) an () tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 2 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.lrate = lrate
        self.loss_fn = loss_fn
        h = 32 # The h determined by the question
        self.model = nn.Sequential(
            nn.Linear(in_size, h),
            nn.ReLU(),
            nn.Linear(h, out_size),
        )
        self.optimizer = optim.Adam(self.model.parameters(), self.lrate)
        # raise NotImplementedError("You need to write this part!")

    # def set_parameters(self, params):
    #     """ Sets the parameters of your network.

    #     @param params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    # def get_parameters(self):
    #     """ Gets the parameters of your network.

    #     @return params: a list of tensors containing all parameters of the network
    #     """
    #     raise NotImplementedError("You need to write this part!")

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        y = self.model(x)
        return y
        
        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) at this timestep as a float
        """
        y_ev = self.forward(x)
        loss = self.loss_fn(y_ev, y) # calculate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        L = loss.item()
        return L
        # raise NotImplementedError("You need to write this part!")
        # return 0.0

def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net. Use the full batch size.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param n_iter: an int, the number of epoches of training
    @param batch_size: size of each batch to train on. (default 100)

    NOTE: This method _must_ work for arbitrary M and N.

    @return losses: array of total loss at the beginning and after each iteration.
            Ensure that len(losses) == n_iter.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    losses = []
    yhats = []
    shape_train = train_set.size()
    shape_dev = dev_set.size()
    dev_size = shape_dev[0]    
    
    lrate = 0.00096
    loss_fn = nn.CrossEntropyLoss()
    in_size = shape_train[1]
    out_size = 2
    #print("The learning rate now is ",lrate)
    
    # Normalize the sets
    net = NeuralNet(lrate, loss_fn, in_size, out_size)
    Train_mean = torch.mean(train_set)
    Train_std = torch.std(train_set)
    train_set_normalize = (train_set-Train_mean)/Train_std
    Dev_mean = torch.mean(dev_set)
    Dev_std = torch.std(dev_set)
    dev_set_normalize = (dev_set-Dev_mean)/Dev_std
    
    # Training the net
    bat_list = torch.split(train_set_normalize, batch_size, dim=0)
    bat_label = torch.split(train_labels, batch_size, dim=0)
    length = len(bat_list)
    for i in range(n_iter):
        loss = net.step(bat_list[i % length], bat_label[i % length])
        losses.append(loss)
        
    # Developing the net    
    for j in range(dev_size):
        dev_bin = net.forward(dev_set_normalize[j])
        if dev_bin[0] > dev_bin[1]:
            yhats.append(0)  
        else:
            yhats.append(1)
    
    return losses, yhats, net
