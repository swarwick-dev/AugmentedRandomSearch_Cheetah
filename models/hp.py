import json
import os

class Hp:
    def __init__(self, **kwargs):
        self.nb_steps = kwargs.get('nb_steps', 1000)
        self.episode_length = kwargs.get('episode_length', 1000)
        self.learning_rate = kwargs.get('learning_rate', 0.02)
        self.nb_directions = kwargs.get('nb_directions', 16)
        self.nb_best_directions = kwargs.get('nb_best_directions', 16)
        assert self.nb_best_directions <= self.nb_directions
        self.noise = kwargs.get('noise', 0.03)
        self.initial_noise = kwargs.get('initial_noise', 0.1)
        self.noise_decay = kwargs.get('noise_decay', 0.99)
        self.seed = kwargs.get('seed', 1)
        self.env_name = kwargs.get('env_name', 'HalfCheetah-v4')
        self.patience = kwargs.get('patience', 10)
        self.min_delta = kwargs.get('min_delta', 0.01)

    @classmethod
    def load_from_file(cls, filename: str) -> 'Hp':
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                return cls(**data)
        else:
            return cls()

    def save_to_file(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)