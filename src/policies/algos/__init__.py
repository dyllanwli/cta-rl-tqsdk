from .ppo import PPOConfig
import gym

from ray.tune.trainable import Trainable


class Algos:
    def __init__(self, name: str, env: gym.Env, env_config):
        if name == "PPO":
            self.algo = PPOConfig(env=env, env_config=env_config)
        else:
            raise ValueError("Not found")

    @property
    def trainer(self) -> Trainable:
        return self.algo.trainer()

    @property
    def config(self):
        return self.algo.config
