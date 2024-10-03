import json
import numpy as np

class BaseModel:
    def __init__(self, input_size, output_size, hp):
        self.theta = np.zeros((output_size, input_size))  # Policy weights
        self.hp = hp

    def evaluate(self, input, delta=None, direction=None):
        raise NotImplementedError

    def sample_deltas(self):
        raise NotImplementedError

    def update(self, rollouts, sigma_r):
        raise NotImplementedError

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({'theta': self.theta.tolist(), 'hp': vars(self.hp)}, f)

    def load(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.theta = np.array(data['theta'])
            self.hp = Hp(**data['hp'])
