import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from collections import deque                                  

pathfile = pathlib.Path(__file__).parent.resolve()
model_path = str(pathfile)

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
action_size = env.action_space.n
state_size = env.observation_space.shape

# Reset the environment
state = env.reset()
action = 1
state = env.step(action)[0] 

done = False
total_reward = 0
total_step = 0
all_rewards = []


# Run the game loop without exploration
#while not done:
for e in range(10):
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        # Take a step in the environment
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_step += 1
    all_rewards.append(total_reward)

print("total_step: {}, score: {}".format(total_step, total_reward))

plt.plot(all_rewards)
plt.ylabel('Total Reward')
plt.show()
env.close()