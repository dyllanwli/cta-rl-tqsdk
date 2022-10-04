from .ppo import PPOConfig
import gym


class Algos:
    def __init__(self, name: str, env: gym.Env, env_config):
        if name == "ppo":
            self.algo = PPOConfig(env=env, env_config=env_config)
        else:
            raise ValueError("Not found")

    def trainer(self):
        return self.algo.trainer()

    def config(self):
        return self.algo.config
