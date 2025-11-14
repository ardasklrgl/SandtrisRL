import sys 
import types
import numpy

import gymnasium as gym
import gymnasium.spaces as spaces
from stable_baselines3 import PPO
import os
from sandtris_env5 import SandtrisEnv
import time

print(f"Using NumPy version: {numpy.__version__}")
print(f"Using Python version: {sys.version}")

'''
Versions are important for testing;
if the model on a different version of np or py, the test will not run.
Make sure the train and test versions match.
Creating a conda env and setting the appropiate versions is an easy solution.
'''

env = SandtrisEnv(render_mode="human")

model_path = "sandtrisRL/models/PPO/best_model.zip"
# Try loading without env first
try:
    model = PPO.load(model_path, env=env, custom_objects={
        'learning_rate': 0.0,
        'lr_schedule': lambda _: 0.0,
        'clip_range': lambda _: 0.2,
    })
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying alternative loading method...")
    
    # Alternative: load without environment
    model = PPO.load(model_path, env=None, device='cpu')
    model.set_env(env)
    print("Model loaded with alternative method!")

for episode in range(10):
    observation, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    while not (done or truncated):
        action, _ = model.predict(observation)
        observation, reward, done, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        time.sleep(0.001)