from .ppo import PPOConfig
from .a3c import A3CConfig
import gym

from ray.rllib.algorithms.algorithm import Algorithm


class Algos:
    def __init__(self, name: str, env: gym.Env, env_config):
        if name == "PPO":
            self.algo = PPOConfig(env=env, env_config=env_config)
        elif name == "A3C":
            self.algo = A3CConfig(env=env, env_config=env_config)
        else:
            raise ValueError("Not found")

    @property
    def trainer(self) -> Algorithm:
        return self.algo.trainer()

    @property
    def config(self):
        return self.algo.config
