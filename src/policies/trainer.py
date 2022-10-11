
import logging
from tabnanny import verbose
logging.getLogger('tensorflow').disabled = True
import wandb
from pprint import pprint

from tqsdk import TqApi, TqAuth, TqBacktest, TqSim
from .envs.constant import EnvConfig
# from .envs import FuturesEnvV2_2 as FuturesEnv
from .envs import FuturesEnvV2_3 as FuturesEnv
from .algos import Algos
from datetime import date, datetime

import gym
from ray import air, tune
from ray.air.result import Result
from ray.air.callbacks.wandb import WandbLoggerCallback
import ray


class RLTrainer:
    def __init__(self, auth: TqAuth):
        print("Initializing RL trainer")
        wandb_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # wandb.init(project="futures-trading", name=wandb_name)

        self.env_config = {"cfg": EnvConfig(
            auth=auth,
            symbols=["cotton"],
            start_dt=date(2016, 1, 1),
            end_dt=date(2022, 3, 1),
            dataloader="db",
            wandb_name=wandb_name
        )}
        self.env = FuturesEnv

        self.cb = [WandbLoggerCallback(
            project="futures-trading",
            log_config=True,
        )]

        ray.init(logging_level=logging.INFO, num_cpus=40, num_gpus=1)

    async def wandb_log(self, result):
        if result:
            wandb.config.update({result['config']})
            for k in result['info'].keys():
                wandb.log({"info/" + k: result['info'][k]})
            wandb.log(
                {"info/num_agent_steps_trained": result['num_agent_steps_trained']})
            for k in result['sampler_perf'].keys():
                wandb.log({"sampler_perf/" + k: result['sampler_perf'][k]})
            for k in result['sampler_results'].keys():
                wandb.log(
                    {"sampler_results/" + k: result['sampler_results'][k]})

    def train(self, algo_name: str = "A3C", type: str = "tune"):
        algos = Algos(name=algo_name, env=self.env,
                      env_config=self.env_config)
        if type == "tune":
            stop = {
                "training_iteration": 1000000,
                "episode_reward_mean": 1000,
            }
            tuner = tune.Tuner(algo_name, param_space=algos.config,
                            run_config=air.RunConfig(
                name = "futures-trading",
                stop=stop,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=100),
                callbacks=self.cb
            ))
            results = tuner.fit()

            metric = "episode_reward_mean"

            best_result: Result = results.get_best_result(metric, mode="max")
            print("Best result:", best_result)
            print("Checkpoints path:", best_result.best_checkpoints)
        else:
            trainer = algos.trainer
            for i in range(1000000):
                result = trainer.train()
                print(pprint(result))
                # self.wandb_log(result)
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
