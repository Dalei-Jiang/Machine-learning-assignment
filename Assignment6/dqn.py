import gym
import numpy as np
import torch
from torch import nn

import utils
from policies import QPolicy

# Modified by Mohit Goyal (mohit@illinois.edu) on 04/20/2022

# TODO: define the network
def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    model = nn.Sequential(torch.nn.Linear(statesize,4), 
                                torch.nn.Sigmoid(),
                                torch.nn.Linear(4,actionsize))
    return model
    
class DQNPolicy(QPolicy):
    """
    Function approximation via a deep network
    """

# TODO: create optimizer and loss
    def __init__(self, model, statesize, actionsize, lr, gamma):
        """
        Inititalize the dqn policy

        @param model: the nn.Module instance returned by make_dqn
        @param statesize: dimension of the input continuous state space.
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate 
        @param gamma: discount factor
        """
        super().__init__(statesize, actionsize, lr, gamma)
        self.statesize = statesize
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma
        if model is None:
            self.model = make_dqn(statesize, actionsize)
        else:
            self.model = model

    def qvals(self, state):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.from_numpy(state).type(torch.FloatTensor)
            qvals = self.model(states)
        return qvals.numpy()

# TODO: temporal difference learing update
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
        self.model.train()
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        state_tensor = torch.tensor(state)
        action_tensor = torch.tensor(action)
        
        loss_function = nn.L1Loss()
        f = self.model(state_tensor)
        y = action_tensor
        loss = loss_function(f,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()**2
        
        
        # if done:
        #     target = reward
        # else:
        #     max_Q_sa = max(self.model[next_state_dc])
        #     target = reward + self.gamma * max_Q_sa # 0 represents an action in state next_state
        # q_loss = target - self.model[state_dc][action]
        # self.model[state_dc][action] += self.lr*q_loss
        
        # loss = q_loss**2
        # return loss # the square error of the original q-value estimate    

        

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
    torch.manual_seed(0) # seed torch
    torch.use_deterministic_algorithms(True) # use deterministic algorithms

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    policy = DQNPolicy(make_dqn(statesize, actionsize), statesize, actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'dqn.model')
