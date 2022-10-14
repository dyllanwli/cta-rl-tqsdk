
from API import API
import logging

from .envs.constant import EnvConfig
from .envs import FuturesEnvV3_1 as FuturesEnv
from .algos import Algos
from pprint import pprint
from tqsdk import TqApi, TqAuth, TqBacktest, TqSim
import gym
from ray import air, tune
from ray.air.result import Result
from ray.air.callbacks.wandb import WandbLoggerCallback
import ray
from datetime import date, datetime


# from .envs import FuturesEnvV2_2 as FuturesEnv


class RLTrainer:
    def __init__(self, account: str = "a4", train_type: str = "train"):
        print("Initializing RL trainer")
        auth = API(account=account).auth
        self.train_type = train_type  # tune or train
        self.algo_name = "IMPALA"

        self.wandb_name = self.algo_name + "_" + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S") if self.train_type == "train" else False
        # only trainer mode will log to wandb in env
        self.env_config = {"cfg": EnvConfig(
            auth=auth,
            symbols=["cotton"],
            # symbols=["sliver"],
            start_dt=date(2016, 1, 1),
            end_dt=date(2021, 3, 1),
            wandb=self.wandb_name,
            is_offline=True,
            is_random_sample=True,
        )}
        self.env = FuturesEnv

        ray.init(logging_level=logging.INFO, num_cpus=60, num_gpus=1)

    def train(self,):
        is_tune = self.train_type == "tune"
        algos = Algos(name=self.algo_name, env=self.env,
                      env_config=self.env_config, is_tune=is_tune)
        if is_tune:
            # use tuner
            stop = {
                "training_iteration": 1000,
                "episode_reward_mean": 1,
            }
            cb = [WandbLoggerCallback(
                project="futures-trading-3",
                group="tune",
                log_config=True,
            )]
            tuner = tune.Tuner(self.algo_name, param_space=algos.config,
                               run_config=air.RunConfig( 
                                   name="futures-trading",
                                   stop=stop,
                                   checkpoint_config=air.CheckpointConfig(
                                       checkpoint_frequency=100),
                                   callbacks=cb
                               ))
            results = tuner.fit()
            metric = "episode_reward_mean"
            best_result: Result = results.get_best_result(metric, mode="max")
            print("Best result:", best_result)
            print("Checkpoints path:", best_result.best_checkpoints)
        else:
            # use trainer
            trainer = algos.trainer
            for i in range(1000000):
                result = trainer.train()
                self.logging(result)
                if i % 500 == 0:
                    checkpoint = trainer.save(checkpoint_dir="checkpoints")
                    print("checkpoint saved at", checkpoint)
        ray.shutdown()

    def run(self, checkpoint_path, max_episodes: int = 1000):
        trainer = Algos(name=self.algo_name, env=self.env,
                        env_config=self.env_config, train_type=self.train_type).trainer
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

    def logging(self, result):
        print("timers", result['timers'])
        print("info", result['info'])
        # print("sampler_results", result['sampler_results'])
        # def wandb_log(result):
        #     wandb.config.update({result['config']})
        #     for k in result['info'].keys():
        #         wandb.log(data={"info/" + k: result['info'][k]})
        #     wandb.log(
        #         data={"info/num_agent_steps_trained": result['num_agent_steps_trained']})
        #     for k in result['sampler_perf'].keys():
        #         wandb.log(data={"sampler_perf/" +
        #                   k: result['sampler_perf'][k]})
        #     for k in result['sampler_results'].keys():
        #         wandb.log(
        #             data={"sampler_results/" + k: result['sampler_results'][k]})

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
