from ray.rllib.algorithms.algorithm import Algorithm
from .ppo import PPOConfig
import gym


class Agent:
    def __init__(self, name: str, env: gym.Env, env_config):
        if name == "ppo":
            self.agent = PPOConfig(env=env, env_config=env_config)
        else:
            raise ValueError("Not found")

    def build(self) -> Algorithm:
        return self.agent.build()

    def config(self):
        return self.agent.config
