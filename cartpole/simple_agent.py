# https://arxiv.org/pdf/1710.02298.pdf
#https://github.com/marload/deep-rl-tf2
# https://github.com/sourcecode369/deep-reinforcement-learning/blob/master/DQN/CartPole-v1/CartPole-v1%20DQN%20with%20Fixed%20Q%20Targets%20.ipynb
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime             

import simplem as am
import pathlib
from collections import deque

    
pathfile = pathlib.Path(__file__).parent.resolve()

env = gym.make("CartPole-v1")
env.reset()
action_size = env.action_space.n
state_size = env.observation_space.shape[0]

print("Action Space {}".format(action_size))
print("State Space {}".format(state_size))

pathfile =  str(pathfile) + r"\cart_5"
dqn = am.DQN_Agent(env, state_size, action_size, pathfile, train = True)

# load model
EPISODES = 10
max_steps = 300
total_step = 0

all_rewards = []
all_Q = []
exploration_rates =[]
all_avg_reward =[]
all_avg_er = []


# First generate training data to store in the memory_buffer as experience so it can be used to sample from to update the Q-model
#dqn.generate_experience()

for episode in range(EPISODES):

    state = env.reset() # start new episode
    state = np.reshape(state, [1, state_size])
    
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = dqn.get_action(state, explore=False)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # state_stack = deque(list(state_stack)[1:] + [next_state], maxlen=4) # Update state stack
        total_reward += reward
        state = next_state
        total_step += 1

    all_rewards.append(total_reward)
    exploration_rates.append(dqn.epsilon)
    avg_reward = sum(all_rewards[-10:]) / len(all_rewards[-10:]) # calculate the average reward over the last 10 episodes
    all_avg_reward.append(avg_reward)
    avg_explore_rate = sum(exploration_rates[-10:]) / len(exploration_rates[-10:]) # calculate the average exploration rate over the last 10 episodes
    all_avg_er.append(avg_explore_rate)
    print("total_step: {}, episode: {}/{}, score: {}, e: {:.2}, avg_reward: {:.2}, avg_explore_rate: {:.2}".format(total_step, episode, EPISODES, total_reward, dqn.epsilon, avg_reward, avg_explore_rate))

plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
env.close()

