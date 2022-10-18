from .ppo import PPOConfig
from .a3c import A3CConfig
from .a2c import A2CConfig
from .sac import SACConfig
from .r2d2 import R2D2Config
from .impala import IMPALAConfig
import gym

from ray.rllib.algorithms.algorithm import Algorithm
class Algos:
    def __init__(self, name: str, env: gym.Env, env_config, is_tune: bool):
        if name == "PPO":
            self.algo = PPOConfig(env=env, env_config=env_config, is_tune = is_tune)
        elif name == "A3C":
            self.algo = A3CConfig(env=env, env_config=env_config, is_tune = is_tune)
        elif name == "A2C":
            self.algo = A2CConfig(env=env, env_config=env_config, is_tune = is_tune)
        elif name == "SAC":
            self.algo = SACConfig(env=env, env_config=env_config, is_tune = is_tune)
        elif name == "R2D2":
            self.algo = R2D2Config(env=env, env_config=env_config, is_tune = is_tune)
        elif name == "IMPALA":
            self.algo = IMPALAConfig(env=env, env_config=env_config, is_tune = is_tune)
        else:
            raise ValueError("Not found")

    @property
    def trainer(self) -> Algorithm:
        return self.algo.trainer()

    @property
    def config(self):
        return self.algo.config
