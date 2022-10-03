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

from .algos import Algos
from .envs.futures_env_v1 import FuturesEnvV1
from .envs.constant import EnvConfig

from tqsdk import TqApi, TqAuth, TqBacktest, TqSim


class RLTrainer:
    def __init__(self, auth: TqAut):

        backtest = TqBacktest(start_dt=date(2021, 1, 1),
                              end_dt=date(2021, 1, 10))

        self.env_config = {"cfg": EnvConfig(
            api=self._set_account(auth, backtest, init_balance=200000),
            symbols=["cotton"],
        )}
        self.env = FuturesEnvV1

        ray.init(logging_level=logging.INFO)

    def _set_account(self, auth, backtest, init_balance=10000000,
                     live_market=False, live_account=None):
        if backtest is not None:
            print("Backtest mode")
            return TqApi(auth=auth, backtest=backtest,
                         account=TqSim(init_balance=init_balance))
        else:
            if live_market:
                print("Live market mode")
                return TqApi(account=live_account, auth=auth)
            else:
                print("Sim mode")
                return TqApi(auth=auth, account=TqSim(
                    init_balance=init_balance))

    def env_creator(self, config):
        return FuturesEnvV1(config)

    def train(self, agent_name: str = "ppo"):
        trainer = Algos(name=agent_name, env=self.env,
                        env_config=self.env_config).build()
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

    def run(self, checkpoint_path, agent_name: str = "ppo", max_episodes: int = 1000):
        trainer = Algos(name=agent_name, env=self.env,
                        env_config=self.env_config).build()
        trainer.restore(checkpoint_path)
        print("Restored from checkpoint path", checkpoint_path)

        env = gym.make(self.env, config=self.env_config)
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
