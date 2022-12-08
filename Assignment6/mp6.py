import argparse

import gym
import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to model')
parser.add_argument('--episodes', type=int, default=8, help='number of episodes')
parser.add_argument('--epsilon', type=float, default=0.01, help='exploration temperature')

args = parser.parse_args()
args.episodes = 8
args.epsilon = 0
args.model = "dqn.model"
print(args)

# Environment (a Markov Decision Process model)
env, statesize, actionsize = gym.make('CartPole-v1'), 4, 2

env.reset(seed=42) # seed the environment
np.random.seed(42) # seed numpy
import random
random.seed(42) # seed random

# Q Model
model = utils.loadmodel(args.model, env, statesize, actionsize)
print("Model: {}".format(model))

# Rollout
_, scores = utils.rollout(env, model, args.episodes, args.epsilon, render=False)

# Report
mean_score, std_score = np.mean(scores), np.std(scores)
print("Score: {:.1f} Mean, {:.3f} STD".format(mean_score, std_score))