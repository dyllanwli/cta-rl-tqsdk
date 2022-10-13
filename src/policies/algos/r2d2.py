from ray.rllib.algorithms import r2d2
import gym
from ray import tune


class R2D2Config:

    def __init__(self, env: gym.Env, env_config):
        self.env = env
        self.config = {
            # basic config 
            "env": env,
            "env_config": env_config,
            "num_workers": 5,
            "num_envs_per_worker": 1,
            "num_cpus_per_worker": 5,
            "num_gpus": 0,
            "framework": "tf",
            "horizon": 14400,  # horizon need to be set
            # R2D2 confi
            "zero_init_states": True,
            "use_h_function": True,
            "h_function_epsilon": 1e-3,
            # DQN overrides
            "adam_epsilon": 1e-3,
            "lr": 1e-5,
            "gamma": tune.grid_search([0.5, 0.95]),
            "train_batch_size": 1024,
            "target_network_update_freq": 1000,
            "training_intensity": 150,
            # R2D2 is using a buffer that stores sequences.
            "replay_buffer_config": {
                "type": "MultiAgentReplayBuffer",
                "capacity": 100000,
                "storage_unit": "sequences",
                "replay_burn_in": 20,
            },
            "model": {
                "fcnet_hiddens": [256, 256, 256],
                "fcnet_activation": "relu",
                "use_lstm": True, # use LSTM or use attention
                "max_seq_len": tune.grid_search([20, 40, 60]),
                "lstm_cell_size": tune.grid_search([256, 512]),
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
    
    def trainer(self) -> r2d2.R2D2:
        config = r2d2.R2D2_DEFAULT_CONFIG.copy()
        config.update(self.config)
        return r2d2.R2D2(env=self.env, config=config)
