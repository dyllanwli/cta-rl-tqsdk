from datetime import date, datetime
from math import gamma
from pprint import pprint
import logging

import wandb
import gym
import ray

from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.algorithms.registry import get_algorithm_class

from .agents import Agent
from .envs.futures_env_v1 import FuturesEnvV1
from .envs.constant import EnvConfig

from tqsdk import TqApi, TqAuth, TqBacktest


class RLTrainer:
    def __init__(self, auth: TqAuth):

        backtest = TqBacktest(start_dt=date(2021, 1, 1),
                              end_dt=date(2021, 1, 10))

        self.env_config = EnvConfig(
            auth=auth,
            symbols=["cotton"],
            backtest=backtest,
            live_market=False,
        )
        self.env_name = "FuturesEnv-v1"

        register_env(self.env_name, lambda config: FuturesEnvV1(config))

        ray.init(logging_level=logging.INFO)

    def train(self, agent: str = "ppo"):
        trainer = Agent(agent).build(env=self.env_name)
        wandb.init(project="futures-trading", name="train_" +
                   datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        for i in range(10000):
            print(f"Training iteration {i}")
            result = trainer.train()
            print(pprint(result))
            if i % 100 == 0:
                checkpoint = trainer.save(checkpoint_dir="checkpoints")
                print("checkpoint saved at", checkpoint)
        ray.shutdown()

    def run(self, checkpoint_path, agent: str = "ppo", max_episodes: int = 1000):
        trainer = Agent(agent).build(env=self.env_name)
        trainer.restore(checkpoint_path)
        print("Restored from checkpoint path", checkpoint_path)

        env = gym.make(self.env_name, config=self.env_config)
        obs = env.reset()

        wandb.init(project="futures-trading", name="run_" +
                   datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        num_episodes = 0
        while num_episodes < max_episodes:
            action = trainer.compute_single_action(obs)

            obs, reward, done, info = env.step(action)
            info["reward"] = reward
            wandb.log(info)
            if done:
                num_episodes += 1
                obs = env.reset()
        ray.shutdown()

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
