import math

import gym
import numpy as np
import torch
import utils
from policies import QPolicy

# Modified by Mohit Goyal (mohit@illinois.edu) on 04/20/2022

class TabQPolicy(QPolicy): # 这一class用作创建一个Q-learning的Policy查找依据
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update 
        @param gamma: discount factor
        @param model (optional): Load a saved table of Q-values for each state-action
            model = np.zeros(self.buckets + (actionsize,))
            
        """
        # TODO: Initialize the env
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env # 引入环境env
        self.buckets = buckets # 将state离散化的依据
        self.actionsize = actionsize # action的自由度，即维度
        self.lr = lr #learning rate, 目前设为1e-4
        self.gamma = gamma #discount factor, γ
        # if model == None:
        if model is None: # array with length > 0 should use "is"
            self.model = np.zeros(self.buckets + (actionsize,))
        else:
            self.model = model # 如果model为None，创建一个新的初始化model
        # print(self.model)
        # input()
        return
        

    def discretize(self, obs): # 用于把连续的state分割成bins，返回格式为tuple
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation  
        """
        # print("The obs is:", obs)
        # input()
        upper_bounds = [self.env.observation_space.high[0], 5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        # print("The discrete obs is:", new_obs)
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action.
        """
        # TODO: return the Q value for the states
        # print("The states are:",states)
        # input()
        qvals = [] # Initialize
        num_state = len(states)
        for i in range(num_state):
            state = states[i]
            state_dc = self.discretize(state) # 根据输出结果，discretize的作用似乎是整数化数据
            qval = self.model[state_dc]
            qvals.append(qval)
        # print("The qvals are:",qvals)
        # input()
        return qvals
        
    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        # TODO: this function is used to updated the model
        # Algorithm:
        # Qt+1(s,a) = Qt(s,a)+a(Qlocal(s,a)-Qt(s,a))
        # Qlocal(s,a) = Rt(s) + gamma*max(Qt(s',a')) action is determined in the function
        # The action is not applied in this function
        loss = 0
        state_dc = self.discretize(state)
        next_state_dc = self.discretize(next_state)
        
        if done:
            target = reward
        else:
            max_Q_sa = max(self.model[next_state_dc])
            target = reward + self.gamma * max_Q_sa # 0 represents an action in state next_state
        q_loss = target - self.model[state_dc][action]
        self.model[state_dc][action] += self.lr*q_loss
        
        loss = q_loss**2
        return loss # the square error of the original q-value estimate
        
    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)
        

if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    env.reset(seed=42) # seed the environment
    np.random.seed(42) # seed numpy
    import random
    random.seed(42)
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(3, 3, 6, 9), actionsize=actionsize, lr=args.lr, gamma=args.gamma)
    utils.qlearn(env, policy, args)
    torch.save(policy.model, 'tabular.npy')
