import os
import numpy as np
import gymnasium as gym
from gymnasium import wrappers

# Hyperparameters
class Hp():
    def __init__(self):
        self.nb_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.nb_directions = 16
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03  # sigma in Gaussian
        self.seed = 1
        self.env_name = 'HalfCheetah-v4'

# Normalizing states        
class Normaliser():
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.smean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.smean.copy()
        self.smean += (x - self.smean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.smean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalise(self, inputs):
        obs_mean = self.smean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

# Building the AI
class Policy():
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))

    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == 'positive':
            return (self.theta + hp.noise * delta).dot(input)
        else:  # 'negative'
            return (self.theta - hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

# Run one episode
def explore(env, normaliser, policy, direction=None, delta=None):
    state, _ = env.reset()
    done = False
    nb_plays = 0
    sum_rewards = 0

    while not done and nb_plays < hp.episode_length:
        normaliser.observe(state)
        state = normaliser.normalise(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, truncated, _ = env.step(action)
        reward = np.clip(reward, -1, 1)
        sum_rewards += reward
        nb_plays += 1

    return sum_rewards

# Train the AI
def train(env, policy, normaliser, hp):
    for step in range(hp.nb_steps):
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions

        # Positive rewards
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normaliser, policy, direction='positive', delta=deltas[k])

        # Negative rewards
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normaliser, policy, direction='negative', delta=deltas[k])

        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()

        # Sort rollouts by reward and select the best directions
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key=lambda x: scores[x])[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

        # Update policy
        policy.update(rollouts, sigma_r)

        # Evaluate the policy after update
        reward_eval = explore(env, normaliser, policy)
        print(f'Step: {step}, Reward: {reward_eval}')

    # Finalize and save video
    env.close()

# Helper to create directories
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Setup directories for video output
monitor_dir = mkdir('..', 'monitor')

# Initialize hyperparameters, environment, and policy
hp = Hp()
np.random.seed(hp.seed)
env = gym.make(hp.env_name, render_mode="rgb_array")
env = wrappers.RecordVideo(env, monitor_dir)

nb_inputs = env.observation_space.shape[0]
nb_outputs = env.action_space.shape[0]

policy = Policy(nb_inputs, nb_outputs)
normaliser = Normaliser(nb_inputs)

# Train the policy
train(env, policy, normaliser, hp)

