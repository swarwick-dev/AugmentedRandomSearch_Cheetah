import numpy as np
from .base_model import BaseModel

class ARSModel(BaseModel):
    def evaluate(self, input, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == 'positive':
            return (self.theta + self.hp.noise * delta).dot(input)
        else:  # 'negative'
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.nb_directions)]

    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += self.hp.learning_rate / (self.hp.nb_best_directions * sigma_r) * step
