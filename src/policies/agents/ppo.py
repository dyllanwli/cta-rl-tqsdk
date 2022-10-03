from ray.rllib.algorithms.ppo import PPO


class PPOConfig:
    """PPO config for futures trading."""

    def __init__(self):
        self.config = {
            "num_workers": 1,
            "framework": "tf2",
            "model": {
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "relu",
            },
        }

    def build(self, env: str) -> PPO:
        return PPO(env=env, config=self.config)
