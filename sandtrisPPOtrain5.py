import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import os
import torch
from sandtris_env5 import SandtrisEnv

# Setup directories
modeldir = "sandtrisRL/models/PPO"
logdir = "sandtrisRL/logs"

# Creates directories if they don't exist
if not os.path.exists(modeldir):
    os.makedirs(modeldir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create vectorized training environment for runninng parallel envs
num_envs = 2 # can be adjusted
vec_env = make_vec_env(SandtrisEnv, n_envs=num_envs)

# Create separate evaluation environment
eval_env = Monitor(SandtrisEnv())

# Setup model
TIMESTEPS = 100000
TRAINING_CYCLES = 5
MODEL_TOTAL_TIMESTEPS = 0

model_path = f"{modeldir}/sandtrisagent_timestep_{MODEL_TOTAL_TIMESTEPS}"
if os.path.exists(model_path + ".zip"):
    model = PPO.load(model_path, env=vec_env)
    print("Loaded existing model. Continuing training...")
else:
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1, 
        tensorboard_log=logdir,
        learning_rate=1e-4,
        n_steps=1024, # rollout length before update
        batch_size=64, # sample size of the collected rollout for SGD
        # (n_steps * n_envs)%batch_size = 0, keep in mind when adjusting
        n_epochs=10,
        gamma=0.98, # discount factor for future rewards
        gae_lambda=0.95,
        clip_range=0.2, # stablises updates, clips policy update ratio
        ent_coef=0.02, # exploration
        vf_coef=0.5,
        max_grad_norm=0.5, # preventing grad explode
        policy_kwargs={"normalize_images": False}, # obs is already normalised
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Created new model.")

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=modeldir,
    n_eval_episodes=5,
    eval_freq=5000,
    verbose=1,
    deterministic=True
)

# Training loop
for i in range(1, TRAINING_CYCLES + 1):
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=eval_callback,
        reset_num_timesteps=False,
        tb_log_name="PPO"
    )
    
    total_ts = TIMESTEPS * i + MODEL_TOTAL_TIMESTEPS
    model.save(f"{modeldir}/sandtrisagent_timestep_{total_ts}")
    print(f"Saved model at {total_ts} timesteps")

# Cleanup
vec_env.close()
eval_env.close()
print("Training complete")