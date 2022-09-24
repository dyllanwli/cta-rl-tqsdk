from pprint import pprint
from .envs.futures_env_v1 import FuturesEnvV1
import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo


class RLTrainer:
    def __init__(self, config):
        self._set_config(config)

        register_env("FuturesEnv-v1", lambda config: FuturesEnvV1(config))

    def _set_config(self, config):
        self.config = config

    def train(self):
        ray.init()
        config = self.config
        config["env"] = "FuturesEnv-v1"
        config["env_config"] = self.config
        config["num_workers"] = 1
        config["num_gpus"] = 0
        config["num_cpus_per_worker"] = 1
        config["num_cpus_for_driver"] = 1
        config["framework"] = "torch"
        config["log_level"] = "DEBUG"
        # config["callbacks"] = {
        #     "on_episode_start": on_episode_start,
        #     "on_episode_step": on_episode_step,
        #     "on_episode_end": on_episode_end,
        #     "on_sample_end": on_sample_end,
        #     "on_train_result": on_train_result,
        #     "on_postprocess_traj": on_postprocess_traj,
        # }
        config["evaluation_interval"] = 1
        config["evaluation_num_episodes"] = 1
        config["evaluation_config"] = {
            "explore": False,
        }

        trainer = ppo.PPOTrainer(config=config)
        for i in range(100):
            print(f"Training iteration {i}")
            result = trainer.train()
            print(pprint(result))

            if i % 10 == 0:
                checkpoint = trainer.save()
                print("checkpoint saved at", checkpoint)

        ray.shutdown()

    def evaluate(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass