"""
Example random agent script using the gymnasium API to demonstrate that procgen works
"""

import gymnasium as gym

env = gym.make('procgen:procgen-coinrun-v0')
obs, info = env.reset()
step = 0
while True:
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    print(f"step {step} reward {rew} done {done}")
    step += 1
    if done:
        break