import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.custom_env import WildlifeEnv

# Setup directories
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)

# Create environment
env = DummyVecEnv([lambda: Monitor(WildlifeEnv(render_mode=None))])

# Optimized hyperparameters for wildlife capture
hyperparams = {
    "learning_rate": 1e-3,  # Increased learning rate for faster adaptation
    "buffer_size": 50000,  # Large buffer for more diverse experiences
    "learning_starts": 5000,  # Start learning earlier
    "batch_size": 64,  # Large batch size
    "tau": 1.0,
    "gamma": 0.95,  # Slightly lower discount factor for more immediate rewards
    "target_update_interval": 500,  # Update target network more frequently
    "train_freq": (4, "step"),
    "gradient_steps": 1,
    "exploration_fraction": 0.3,  # Longer exploration period
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01,  # Lower final epsilon for more exploitation
    "policy_kwargs": dict(
        net_arch=[256, 128],  # Slightly simpler network
    )
}

model = DQN(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/dqn",
    **hyperparams
)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="models/dqn",
    name_prefix="dqn_wildlife",
    save_replay_buffer=True,
    save_vecnormalize=True
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="models/dqn",
    log_path="logs/dqn",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Training
model.learn(
    total_timesteps=100000,  # Increased training duration
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True,
    tb_log_name="dqn_run"
)

model.save("models/dqn/dqn_wildlife_final")
env.close()
