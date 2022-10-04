from tqsdk import TqApi, TqAuth, TqBacktest, TqSim
from .envs.constant import EnvConfig
from .envs import FuturesEnvV2 as FuturesEnv
from .algos import Algos
from datetime import date, datetime
from pprint import pprint

import gym
import ray

import logging
logging.getLogger('tensorflow').disabled = True

class RLTrainer:
    def __init__(self, auth: TqAuth):

        backtest = TqBacktest(start_dt=date(2021, 1, 1),
                              end_dt=date(2021, 3, 1))

        self.env_config = {"cfg": EnvConfig(
            auth=auth,
            symbols=["cotton"],
            backtest=backtest,
        )}
        self.env = FuturesEnv

        ray.init(logging_level=logging.INFO, num_cpus=10, num_gpus=1)

    def train(self, agent_name: str = "ppo"):
        trainer = Algos(name=agent_name, env=self.env,
                        env_config=self.env_config).trainer()
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
                        env_config=self.env_config).trainer()
        trainer.restore(checkpoint_path)
        print("Restored from checkpoint path", checkpoint_path)

        env = gym.make(self.env, config=self.env_config)
        obs = env.reset()

        num_episodes = 0
        while num_episodes < max_episodes:
            action = trainer.compute_single_action(obs)

            obs, reward, done, info = env.step(action)
            info["reward"] = reward
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
