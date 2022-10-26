
import logging

from .envs.constant import EnvConfig
from .envs import FuturesEnvV3_1 as FuturesEnv
from .algos import Algos
from pprint import pprint
import gym
from ray import air, tune
from ray.air.result import Result
from ray.air.callbacks.wandb import WandbLoggerCallback
import ray
from datetime import date, datetime

from utils.utils import Interval, max_step_by_day
from utils.api import API
# from .envs import FuturesEnvV2_2 as FuturesEnv

class RLTrainer:
    def __init__(self, account: str = "a4", train_type: str = "tune"):
        print("Initializing RL trainer")
        auth = API(account=account).auth
        self.train_type = train_type  # tune or train
        self.algo_name = "A3C"

        self.wandb_name = self.algo_name + "_" + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S") if self.train_type == "train" else False
        self.project_name = "futures-alpha-7"
        INTERVAL = Interval()
        self.interval = INTERVAL.ONE_SEC
        self.max_steps = max_step_by_day[self.interval]
        self.training_iteration = dict({
            INTERVAL.ONE_MIN: 100,
            INTERVAL.FIVE_SEC: 400,
            INTERVAL.ONE_SEC: 500,
        })

        # only trainer mode will log to wandb in env
        self.env_config = {"cfg": EnvConfig(
            auth=auth,
            symbols=["cotton"],
            # symbols=["sliver"],
            start_dt=date(2016, 1, 1),
            end_dt=date(2022, 8, 1),
            wandb=self.wandb_name,
            is_offline=True,
            is_random_sample=True,
            project_name=self.project_name,
            interval=self.interval,
            max_steps=self.max_steps,
            high_freq=True,
        )}
        self.env = FuturesEnv

        ray.init(logging_level=logging.INFO, num_cpus=62, num_gpus=1, include_dashboard=False)

    def train(self,):
        is_tune = self.train_type == "tune"
        algos = Algos(name=self.algo_name, env=self.env,
                      env_config=self.env_config, is_tune=is_tune)
        if is_tune:
            # use tuner
            stop = {
                "training_iteration": self.training_iteration[self.interval],
                "episode_reward_min": 1,
            }
            cb = [WandbLoggerCallback(
                project=self.project_name,
                group="tune_" + self.interval,
                log_config=True,
            )]
            tuner = tune.Tuner(self.algo_name, param_space=algos.config,
                               run_config=air.RunConfig(
                                    verbose=1,
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
            print(algos.config)
            for i in range(self.training_iteration[self.interval]*10):
                result = trainer.train()
                self.logging(result)
                if i % 500 == 0:
                    print(pprint(result))
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

        step = 0
        while step < max_episodes:
            action = trainer.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            info["reward"] = reward
            if done:
                step += 1
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
