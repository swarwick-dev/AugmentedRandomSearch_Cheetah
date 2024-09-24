# pip install gym==0.10.5
#pip install pybullet==2.0.8
#conda install -c conda-forge ffmpeg

import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# Hyper parameters
class Hp():

    def __init__(self):
        self.nb_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.nb_directions = 16
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03 #  sigma in gausian
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0'
        self.learning_rate = 1

# Normalising states        
class Normaliser():

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs) # vector with the number of inputs as size all set to zero
        self.smean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        # Online mean computation
        last_mean = self.smean.copy()
        self.smean += (x - self.smean) / self.n

        # Online variance computation
        self.mean_diff += (x - last_mean) * (x - self.smean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2) # ensures value does not equal zero

    def normalise(self, inputs):
        obs_mean = self.smean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

# Building the AI

class Policy():

    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size)) # matrix of weights, output size = lines, input_size = columns 

    def evaluate(self, 
                 input, 
                 delta = None, # the changes default no delta
                 direction = None # +ve, -ve, opposite
                 ):
        # Handle perturbations
        if direction is None:
            return self.theta.dot(input)
        elif direction == 'positive':
            return (self.theta + hp.noise*delta).dot(input)
        else:      
            return (self.theta - hp.noise*delta).dot(input)
        
    def sample_deltas(self):
        return [np.random.rand(*self.theta.shape) for _ in range(hp.nb_directions)] # gets the size of theta x & y and returns as a list
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

# Run one episode
def explore(env, normaliser, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    nb_plays = 0.
    sum_rewards = 0
        
    while not done and nb_plays < hp.episode_length:
        normaliser.observe(state)
        state = normaliser.normalise(state)
        action = policy.evaluate(state, delta, direction)      
        state, reward, done, _ = env.step(action)
        reward = max(min(reward,1), -1) #  max out to 1 and min out to -1 to prevent outlier large rewards
        sum_rewards += reward
        nb_plays += 1
        
    return sum_rewards

# train the ai
def train(env, policy, normaliser, hp):
    
    for step in range(hp.nb_steps):
        
        #  Step 1 - Initialise perturbations deltas and rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        
        #  Step 2 - Get positive rewards
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normaliser, policy, direction = 'positive', delta = deltas[k])
    
        #  Step 3 - Get negative rewards
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normaliser, policy, direction = 'negative', delta = deltas[k])

        #  Step 4 - Gather all rewards and compute std deviation
        all_rewards = np.array(positive_rewards + negative_rewards) #  gives a 32 element array
        sigma_r = all_rewards.std()

        #  Step 5 - Generate the rollouts
        scores = {k:max(r_pos, r_neg) for k,(r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x])[0:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        #  Step 6 - update policy (ai)
        policy.update(rollouts, sigma_r)
        
        #  Step 7 - Print final reward post update
        reward_eval = explore(env, normaliser, policy)
        print('Step: ', step, 'Reward: ', reward_eval)
        

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path




work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)

env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force = True)

nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]

policy = Policy(nb_inputs, nb_outputs)
normaliser = Normaliser(nb_inputs)

train(env, policy, normaliser, hp)
