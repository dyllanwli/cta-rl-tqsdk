from ray.rllib.algorithms import impala
import gym
from ray import tune

class IMPALAConfig:
    def __init__(self, env: gym.Env, env_config, is_tune: bool):
        self.env = env
        self.config = {
            # basic config 
            "env": env,
            "env_config": env_config,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            # "num_cpus_per_worker": 20,
            "num_gpus": 1,
            "framework": "tf",
            "horizon": 14400,  # horizon need to be set
            "train_batch_size": 200, # shoule be >= rollout_fragment_length
            # IMPALA config
            "vtrace": True,
            "vtrace_clip_rho_threshold": 1.0,
            "vtrace_clip_pg_rho_threshold": 1.0,
            "vtrace_drop_last_ts": False,
            # "num_multi_gpu_tower_stacks": 1,
            "minibatch_buffer_size": 128,
            "num_sgd_iter": 5,
            "replay_proportion": 100,
            "replay_buffer_num_slots": tune.grid_search([10, 100, 500]) if is_tune else 100,
            # "learner_queue_size": 16,
            # "learner_queue_timeout": 300,
            # "max_requests_in_flight_per_sampler_worker": 2,
            # "max_requests_in_flight_per_aggregator_worker": 2,
            "timeout_s_sampler_manager": 0.0,
            "timeout_s_aggregator_manager": 0.0,
            "broadcast_interval": 1,
            "num_aggregation_workers": 0,
            "grad_clip": 40.0,
            "opt_type": "adam",
            "lr_schedule": None,
            "decay": 0.99,
            "momentum": 0.0,
            "epsilon": 0.1,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "entropy_coeff_schedule": None,
            "_separate_vf_optimizer": False,
            "_lr_vf": 0.0005,
            "after_train_step": None,
            "lr": 0.00001,
            # "lr_schedule": [[0, 0.00001], [1000, 0.000005]],
            "rollout_fragment_length": 50,
            "min_time_s_per_iteration": 100,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "use_lstm": True, # use LSTM or use attention
                "max_seq_len": 100,
                "lstm_cell_size": 512,
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
    
    def trainer(self) -> impala.Impala:
        config = impala.DEFAULT_CONFIG.copy()
        config.update(self.config)
        return impala.Impala(env=self.env, config=config)
