from ray.rllib.algorithms import sac
import gym


class SACConfig:
    """A3C config for futures trading."""

    def __init__(self, env: gym.Env, env_config):
        self.env = env
        self.config = {
            # basic config
            "env": env,
            "env_config": env_config,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            # "num_cpus_per_worker": 10,
            "num_gpus": 0,
            "framework": "torch",
            "horizon": 1000000,  # horizon need to be set
            # SAC config
            "twin_q": True,
            "q_model_config": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
            },
            "policy_model_config": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
            },
            "clip_actions": True,
            "tau": 0.005,
            "initial_alpha": 0.5,
            "target_entropy": "auto",
            "n_step": 1,
            "replay_buffer_config": {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": int(1e6),
                # How many steps of the model to sample before learning starts.
                "learning_starts": 1500,
                # If True prioritized replay buffer will be used.
                "prioritized_replay": False,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                # Whether to compute priorities already on the remote worker side.
                "worker_side_prioritization": False,
            },
            "store_buffer_in_checkpoints": False,
            "training_intensity": None,
            "optimization": {
                "actor_learning_rate": 0.00001,
                "critic_learning_rate": 0.00001,
                "entropy_learning_rate": 0.00001,
            },
            "grad_clip": None,  # 40.0
            "target_network_update_freq": 0,
            # .rollout
            "rollout_fragment_length": 20,
            "compress_observations": False,
            # .trainig
            "train_batch_size": 256, # shoule be >= rollout_fragment_length
            # .reporting
            "min_time_s_per_iteration": 1,
            "min_sample_timesteps_per_iteration": 100,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "use_lstm": False,  # use LSTM or use attention
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

    def trainer(self) -> sac.SAC:
        config = sac.DEFAULT_CONFIG.copy()
        config.update(self.config)
        return sac.SAC(env=self.env, config=config)
