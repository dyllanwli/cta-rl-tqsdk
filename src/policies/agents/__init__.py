from ray.rllib.algorithms.algorithm import Algorithm
from .ppo import PPOConfig


class Agent:
    def __init__(self, name: str):
        if name == "ppo":
            self.agent = PPOConfig()
        else:
            raise ValueError("Not found")

    def build(self, env: str) -> Algorithm:
        return self.agent.build(env=env)

    def config(self):
        return self.agent.config
