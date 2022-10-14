from ray.rllib.algorithms import ppo
import gym
from ray import tune

class PPOConfig:

    def __init__(self, env: gym.Env, env_config, is_tune: bool):
        self.env = env
        self.config = {
            "env": env,
            "env_config": env_config,
            "num_workers": 5,
            "num_envs_per_worker": 1,
            # "num_cpus_per_worker": 20,
            "num_gpus": 1,
            "framework": "tf",
            "horizon": 14400,  # horizon need to be set
            "use_gae": True,
            "clip_param": 0.3,
            "lambda": tune.grid_search([0.4, 0.5, 0.99]) if is_tune else 0.4,
            "sgd_minibatch_size": 128,
            "lr": 0.00001,
            "lr_schedule": [[0, 0.00001], [1000, 0.000005]],
            "vf_loss_coeff": 0.5,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "use_lstm": True, # use LSTM or use attention
                "max_seq_len": 50,
                "lstm_cell_size": 256,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": False,
                "_time_major": True,
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
    
    def trainer(self) -> ppo.PPO:
        config = ppo.DEFAULT_CONFIG.copy()
        config.update(self.config)
        return ppo.PPO(env=self.env, config=config)
