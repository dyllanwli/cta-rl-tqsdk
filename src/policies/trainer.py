import logging
logging.getLogger('tensorflow').disabled = True

import ray
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray import air, tune
import gym
from pprint import pprint
from datetime import date, datetime
from .algos import Algos
from .envs import FuturesEnvV2_2 as FuturesEnv
from .envs.constant import EnvConfig
from tqsdk import TqApi, TqAuth, TqBacktest, TqSim

import wandb



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
            wandb_name = wandb_name
        )}
        self.env = FuturesEnv

        self.cb = [WandbLoggerCallback(
            project="futures-trading",
            name=wandb_name,
            log_config=True,
        )]

        ray.init(logging_level=logging.INFO, num_cpus=20, num_gpus=1)
    
    async def wandb_log(self, result):
        if result:
            wandb.config.update({result['config']})
            for k in result['info'].keys():
                wandb.log({"info/" + k: result['info'][k]})
            wandb.log({"info/num_agent_steps_trained": result['num_agent_steps_trained']})
            for k in result['sampler_perf'].keys():
                wandb.log({"sampler_perf/" + k: result['sampler_perf'][k]})
            for k in result['sampler_results'].keys():
                wandb.log({"sampler_results/" + k: result['sampler_results'][k]})

    def train(self, agent_name: str = "ppo"):
        algos = Algos(name=agent_name, env=self.env,
                        env_config=self.env_config)
        stop = {
            "training_iteration": 1000000,
            "episode_reward_mean": 1000,
        }
        tuner = tune.Tuner(algos.trainer(), param_space=algos.config, 
            run_config=air.RunConfig(
                stop = stop, 
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=100),
                callbacks=self.cb
        ))
        results = tuner.fit()

        metric = "episode_reward_mean"

        best_result = results.get_best_result(metric, mode="max")
        print("Best result:", best_result)
        print("Checkpoint path:", best_result["checkpoint"])

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
