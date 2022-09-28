from datetime import date
from pprint import pprint
import logging

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.tune.integration.wandb import WandbLoggerCallback

from .envs.futures_env_v1 import FuturesEnvV1
from .envs.constant import EnvConfig

from tqsdk import TqApi, TqAuth, TqBacktest



class RLTrainer:
    def __init__(self, auth: TqAuth):

        backtest = TqBacktest(start_dt=date(2021, 1, 1), end_dt=date(2021, 1, 10))

        self.env_config = EnvConfig(
            auth=auth,
            symbol_name="cotton",
            backtest = backtest,
            initial_cash=200000,
            live_market=False,
        )
        
        register_env("FuturesEnv-v1", lambda config: FuturesEnvV1(config))

    def train(self):
        ray.init(logging_level=logging.INFO)
        config = ppo.DEFAULT_CONFIG.copy()
        config["env"] = "FuturesEnv-v1"
        config["env_config"] = self.config
        config["num_workers"] = 1
        config["num_gpus"] = 0
        config["num_cpus_per_worker"] = 1
        config["num_cpus_for_driver"] = 1
        config["framework"] = "torch"
        config["log_level"] = "DEBUG"
        config["callbacks"] = WandbLoggerCallback(
            project="tqrl-dev",
            log_config=True,
        )
        
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