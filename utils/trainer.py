import logging
import numpy as np
import gymnasium as gym
from gymnasium import wrappers
from torch.utils.tensorboard import SummaryWriter
from utils.normaliser import Normaliser
from models.ars_model import ARSModel
from models.base_model import BaseModel
from models.hp import Hp
from typing import Optional, Callable
import os
import datetime

class Trainer:
    def __init__(self, env_name: str, model_name: str, hp: Optional[Hp], base_dir: str, update_queue: Optional[Callable[[str], None]], video_interval: int):
        self.env_name = env_name
        self.model_name = model_name
        self.hp = hp if hp else Hp(env_name=env_name)
        self.base_dir = base_dir
        self.update_queue = update_queue
        self.video_interval = video_interval

        # Setup directories for video output and TensorBoard logs
        self.monitor_dir = self.mkdir(base_dir, 'videos')
        self.log_dir = self.mkdir(base_dir, 'logs')
        self.checkpoint_dir = self.mkdir(base_dir, 'checkpoints')

        # remove old tensortboard logs
        for file in os.listdir(self.log_dir):
            if file.startswith("events"):
                os.remove(os.path.join(self.log_dir, file))

        # remove old checkpoints
        for file in os.listdir(self.checkpoint_dir):
            os.remove(os.path.join(self.checkpoint_dir, file))

        np.random.seed(self.hp.seed)

        # Initialize environment with video recording and set up TensorBoard writer
        self.env = gym.make(self.hp.env_name, render_mode="rgb_array")
        self.env = wrappers.RecordVideo(self.env, self.monitor_dir, episode_trigger=lambda episode_id: episode_id % self.video_interval == 0, 
                                        disable_logger=True, video_length=30, name_prefix=f'video_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')
        self.writer = SummaryWriter(self.log_dir)

        # Initialize normaliser and model (policy)
        nb_inputs = self.env.observation_space.shape[0]
        nb_outputs = self.env.action_space.shape[0]

        self.normaliser = Normaliser(nb_inputs)
        self.model = ARSModel(nb_inputs, nb_outputs, self.hp)

    def mkdir(self, base_dir: str, name: str) -> str:
        path = os.path.join(base_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    def explore(self, direction=None, delta=None) -> float:
        state, _ = self.env.reset()
        done = False
        nb_plays = 0
        sum_rewards = 0

        while not done and nb_plays < self.hp.episode_length:
            self.normaliser.observe(state)
            state = self.normaliser.normalise(state)
            action = self.model.evaluate(state, delta, direction)
            state, reward, done, truncated, _ = self.env.step(action)
            reward = np.clip(reward, -1, 1)
            sum_rewards += reward
            nb_plays += 1

        return sum_rewards

    def train(self) -> None:
        best_reward = -float('inf')
        patience_counter = 0

        for step in range(self.hp.nb_steps):
            self.hp.noise = max(0.01, self.hp.initial_noise * (self.hp.noise_decay ** step))  # Dynamic noise scaling

            deltas = self.model.sample_deltas()
            positive_rewards = [self.explore('positive', deltas[k]) for k in range(self.hp.nb_directions)]
            negative_rewards = [self.explore('negative', deltas[k]) for k in range(self.hp.nb_directions)]
            all_rewards = np.array(positive_rewards + negative_rewards)
            sigma_r = all_rewards.std()

            # Sort rollouts by reward and select best directions
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x])[:self.hp.nb_best_directions]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Update policy
            self.model.update(rollouts, sigma_r)

            # Evaluate the model after update
            reward_eval = self.explore()
            self.writer.add_scalar('Reward', reward_eval, step)
            self.writer.add_scalar('Noise', self.hp.noise, step)

            # Early stopping
            """ if reward_eval > best_reward + self.hp.min_delta and reward_eval/self.hp.episode_length > 0.9:
                best_reward = reward_eval
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.hp.patience:
                if self.update_queue:
                    self.update_queue.put(f'Early stopping at step {step} due to minimal improvement.')
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{self.hp.env_name}_step_final.json')
                self.model.save(checkpoint_path)
                break """

            # Save model checkpoint every 100 steps
            if step % 100 == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{self.hp.env_name}_step_{step}.json')
                self.model.save(checkpoint_path)

            if self.update_queue:
                self.update_queue.put(f'Step: {step}, Reward: {reward_eval}')

        self.env.close()
        self.writer.close()

        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{self.hp.env_name}_step_final.json')
        self.model.save(checkpoint_path)