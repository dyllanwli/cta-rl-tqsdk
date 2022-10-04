from ray.rllib.algorithms import ppo
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
            "framework": "tf",
            "horizon": 1000000, # horizon need to be set 
            "use_gae": True,
            "lr": 0.00001,
        }

    def trainer(self) -> ppo.PPO:
        config = ppo.DEFAULT_CONFIG.copy()
        config.update(self.config)
        return ppo.PPO(env=self.env, config=config)
