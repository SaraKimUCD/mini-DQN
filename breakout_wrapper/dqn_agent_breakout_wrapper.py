import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import numpy as np
import dqn_model as am
import matplotlib.pyplot as plt
import pathlib
from collections import deque
from stable_baselines3.common.atari_wrappers import AtariWrapper                                     

pathfile = pathlib.Path(__file__).parent.resolve()
model_path = str(pathfile)
model_names= [r"\wrapper_10",r"\wrapper_100", r"\wrapper_200", r"\wrapper_300"]

env = gym.make("BreakoutNoFrameskip-v4",render_mode = "human")
env = AtariWrapper(env)
action_size = env.action_space.n
state_size = env.observation_space.shape

print("Action Space {}".format(action_size))
print(env.env.get_action_meanings())
print("State Space {}".format(state_size))
print(env.observation_space.shape)

dqn = am.DQN_Agent(env, state_size, action_size, model_path+model_names[0], trained = True)

max_steps = 500
total_step = 0
all_rewards = []
exploration_rates =[]

for mn in model_names:
    # Reset the environment
    state_stack = []
    dqn.load_my_model(model_path+mn)
    dqn._reset()
    state = env.reset() # start new episode
    # action = 1
    state = env.reset()[0] 
    #state = dqn.preprocess_image(state)
    state = state.reshape(84,84)
    # Need to stack 4 frames together
    state_stack = np.stack([state for _ in range(4)], axis=-1)
    #dqn.see_frames(state_stack)
    done = False
    total_reward = 0
    s = 0

    # Run the game loop without exploration
    #while not done:
    while s in range(max_steps) or not done:
        if total_step % 100 == 0:
            action = 1
        else:
            action = dqn.get_action(state_stack, explore=False)
        
        # Take a step in the environment
        next_state, reward, done, truncated, info  = env.step(action)
        
        # if total_step > 200:
            # plt.imshow(next_state)
            # plt.show()
            # dqn.see_frames(next_state_stack)
        
        # Add new frame, remove oldest one
        #next_state = dqn.preprocess_image(next_state)
        next_state = next_state.reshape(84,84)            
        next_state_stack = np.concatenate([state_stack[..., 1:], next_state[..., np.newaxis]], axis=-1)

        # Update the state
        state_stack = next_state_stack
        total_reward += reward
        total_step += 1
        s+=1
        

    avg_reward = sum(all_rewards[-10:]) / len(all_rewards[-10:]) # calculate the average reward over the last 10 episodes
    avg_explore_rate = sum(exploration_rates[-10:]) / len(exploration_rates[-10:]) # calculate the average exploration rate over the last 10 episodes
    print("total_step: {}, Model Name: {}, score: {}, avg_reward: {:.2}".format(total_step, mn, total_reward, avg_reward))

    
plt.plot(model_names,all_rewards)
plt.xlabel('Model Name')
plt.ylabel('Total Reward')
plt.show()
env.close()