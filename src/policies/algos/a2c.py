from ray.rllib.algorithms import a2c
import gym

class A2CConfig:

    def __init__(self, env: gym.Env, env_config, is_tune: bool):
        self.env = env
        self.config = {
            # basic config 
            "env": env,
            "env_config": env_config,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "num_cpus_per_worker": 20,
            "num_gpus": 1,
            "framework": "tf",
            "horizon": 14400,  # horizon need to be set
            "train_batch_size": 200, # shoule be >= rollout_fragment_length
            # A2C config
            "microbatch_size": 200,
            # A3C config
            "use_critic": True,
            "use_gae": True,
            "lambda": 0.4,
            "grad_clip": 40.0,
            "lr": 0.00001,
            "lr_schedule": [[0, 0.00001], [1000, 0.000005]],
            "vf_loss_coeff": 0.5,
            "rollout_fragment_length": 50,
            "min_time_s_per_iteration": 100,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "use_lstm": True, # use LSTM or use attention
                "max_seq_len": 50,
                "lstm_cell_size": 256,
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
    
    def trainer(self) -> a2c.A2C:
        config = a2c.A2C_DEFAULT_CONFIG.copy()
        config.update(self.config)
        return a2c.A2C(env=self.env, config=config)
