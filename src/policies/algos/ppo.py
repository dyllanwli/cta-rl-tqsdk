from ray.rllib.algorithms import ppo
import gym

class PPOConfig:

    def __init__(self, env: gym.Env, env_config):
        self.env = env
        self.config = {
            "env": env,
            "env_config": env_config,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "num_cpus_per_worker": 20,
            "num_gpus": 1,
            "framework": "tf",
            "horizon": 1000000,  # horizon need to be set
            "use_gae": True,
            "clip_param": 0.3,
            "lambda": 0.999,
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
