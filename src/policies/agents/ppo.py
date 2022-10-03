from ray.rllib.algorithms.ppo import PPO
import gym


class PPOConfig:
    """PPO config for futures trading."""

    def __init__(self, env: gym.Env, env_config):
        self.env = env
        self.config = {
            "env": env,
            "env_config": env_config,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "num_gpus": 1,
            "framework": "tf2",
            "model": {
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            },
        }

    def build(self) -> PPO:
        return PPO(env=self.env, config=self.config)
