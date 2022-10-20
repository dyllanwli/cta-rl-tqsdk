from ray.rllib.algorithms import a3c
import gym
from ray import tune

class A3CConfig:
    def __init__(self, env: gym.Env, env_config, is_tune: bool):
        self.env = env
        self.config = {
            # basic config 
            "env": env,
            "env_config": env_config,
            "num_workers": 8 if is_tune else 1,
            "num_envs_per_worker": 1,
            # "num_cpus_per_worker": 20,
            "num_gpus": 0 if is_tune else 1,
            "framework": "torch",
            "horizon": 14400,  # horizon need to be set
            "train_batch_size": 256, # shoule be >= rollout_fragment_length
            # "simple_optimizer": True,
            # A3C config
            "use_critic": True,
            "use_gae": True,
            "lambda": 0.5,
            "grad_clip": 40.0,
            "lr": 1e-05,
            # "lr_schedule": [[0, 1e-05], [100, 5e-05]],
            "vf_loss_coeff": 0.5,
            "entropy_coeff": tune.grid_search([0.001, 0.0001]) if is_tune else 0.01,
            "rollout_fragment_length": 200,
            "min_time_s_per_iteration": 100,
            "model": {
                "fcnet_hiddens": tune.grid_search([
                    [256, 256, 256]
                ]) if is_tune else [256, 256, 256],
                "fcnet_activation": "relu",
                "use_lstm": tune.grid_search([True, False]) if is_tune else False, # use LSTM or use attention
                "max_seq_len": 50,
                "lstm_cell_size": tune.grid_search([64]) if is_tune else 256,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": False,
                "_time_major": True,
                # attention
                "use_attention": False,
                "attention_num_transformer_units": 1,
                "attention_dim": 64,
                "attention_memory_inference": 10,
                "attention_memory_training": 10,
                "attention_num_heads": 1,
                "attention_head_dim": 64,
                "attention_position_wise_mlp_dim": 64,
            },
        }
    
    def trainer(self) -> a3c.A3C:
        config = a3c.DEFAULT_CONFIG.copy()
        config.update(self.config)
        return a3c.A3C(env=self.env, config=config)
