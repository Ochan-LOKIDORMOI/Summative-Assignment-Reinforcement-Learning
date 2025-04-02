import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from environment.custom_env import WildlifeEnv

# Setup directories
os.makedirs("models/ppo", exist_ok=True)
os.makedirs("logs/ppo", exist_ok=True)

# Create environment
env = DummyVecEnv([lambda: Monitor(WildlifeEnv(render_mode=None))])

# Optimized hyperparameters for wildlife capture
hyperparams = {
    "learning_rate": 1e-4,  # Adjusted learning rate
    "n_steps": 1024,  # Smaller n_steps for more frequent updates
    "batch_size": 64,  # Smaller batch size
    "n_epochs": 10,
    "gamma": 0.97,  # Slightly lower discount factor
    "gae_lambda": 0.92,  # Lower GAE lambda for less variance
    "clip_range": 0.15,  # Tighter clip range
    "clip_range_vf": None,
    "ent_coef": 0.02,  # Adjusted entropy coefficient
    "vf_coef": 0.5,
    "max_grad_norm": 0.8,
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),  # Simplified networks
        activation_fn=torch.nn.ReLU,
        ortho_init=True
    )
}

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/ppo",
    **hyperparams
)

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="models/ppo",
    name_prefix="ppo_wildlife",
    save_replay_buffer=True,
    save_vecnormalize=True
)

eval_callback = EvalCallback(
    env,
    best_model_save_path="models/ppo/",
    log_path="logs/ppo",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Training
model.learn(
    total_timesteps=100000,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True,
    tb_log_name="ppo_run"
)

model.save("models/ppo/ppo_wildlife_final")
env.close()
