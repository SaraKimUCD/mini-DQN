import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

import dqn_model as am
from stable_baselines3.common.atari_wrappers import AtariWrapper
import pathlib
from collections import deque
    
pathfile = pathlib.Path(__file__).parent.resolve()

env = gym.make("BreakoutNoFrameskip-v4")
env = AtariWrapper(env)

obs, info = env.reset()
action_size = env.action_space.n
state_size = env.observation_space.shape

print("Action Space {}".format(action_size))
print(env.env.get_action_meanings())
print("State Space {}".format(state_size))
print(env.observation_space.shape)

pathfile =  str(pathfile) + r"\model"
dqn = am.DQN_Agent(env, state_size, action_size, pathfile, trained = True)

EPISODES = 51
max_steps = 5000
total_step = 0
all_rewards = []
all_Q = []
exploration_rates =[]
all_avg_reward =[]

for episode in range(EPISODES):
    state_stack = []
    dqn._reset()
    full_path = str(pathfile) + r"\wrapper_" + str(episode)
    state = env.reset()[0]
    state = state.reshape(84,84)
    #state = dqn.preprocess_image(state)
    # Need to stack 4 frames together
    state_stack = np.stack([state for _ in range(4)], axis=-1)
    
    done = False
    total_reward = 0
    s=0

    while s in range(max_steps):
        action = dqn.get_action(state_stack, explore=True)

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)
        
        # Add new frame, remove oldest one
        # next_state = dqn.preprocess_image(next_state)
        next_state = next_state.reshape(84,84)
        next_state_stack = np.concatenate([state_stack[..., 1:], next_state[..., np.newaxis]], axis=-1)

        #Store the experience
        dqn.store_experience(state_stack, action, reward, next_state_stack, done)
        total_reward += reward

        all_Q.append(dqn.train_experience(full_path))

        total_step += 1
        s+=1
        # Update the state
        state_stack = next_state_stack

    all_rewards.append(total_reward)
    exploration_rates.append(dqn.epsilon)
    avg_reward = sum(all_rewards[-10:]) / len(all_rewards[-10:]) # calculate the average reward over the last 10 episodes
    all_avg_reward.append(avg_reward)
    avg_explore_rate = sum(exploration_rates[-10:]) / len(exploration_rates[-10:]) # calculate the average exploration rate over the last 10 episodes
    print("total_step: {}, episode: {}/{}, score: {}, e: {:.2}, avg_reward: {:.2}, avg_explore_rate: {:.2}".format(total_step, episode, EPISODES, total_reward, dqn.epsilon, avg_reward, avg_explore_rate))
    
    if episode % 5 == 0:
        # dqn.save_my_model(full_path)
        Q_df = pd.DataFrame(all_Q)
        rewards_df = pd.DataFrame(all_rewards)
        avgr_df = pd.DataFrame(all_avg_reward)

        Q_df.to_csv(str(pathfile)  +'/wrapper_all_Q_' + datetime.today().strftime('%Y%m%d') +'.csv', mode='a', header = False, index= False)
        rewards_df.to_csv(str(pathfile) +'/wrapper_all_rewards_' + datetime.today().strftime('%Y%m%d') +'.csv', mode='a',header = False, index= False)
        avgr_df.to_csv(str(pathfile) +'/wrapper_all_avg_reward_' + datetime.today().strftime('%Y%m%d') +'.csv', mode='a',header = False, index= False)
        
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
env.close()

