import gymnasium as gym
import tensorflow as tf
from tensorflow import keras

import numpy as np
import dqn_model_agent as am
import matplotlib.pyplot as plt
import pathlib
from collections import deque                                  

pathfile = pathlib.Path(__file__).parent.resolve()
model_path = str(pathfile)
# model_names= [r"\dqn_breakout_10",r"\dqn_breakout_500", r"\dqn_breakout_1000", r"\dqn_breakout_1500"]
model_names= [r"\dqn_breakout_demo"]


env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
action_size = env.action_space.n
state_size = env.observation_space.shape

print("Action Space {}".format(action_size))
print(env.env.get_action_meanings())
print("State Space {}".format(state_size))
print(env.observation_space.shape)

dqn = am.DQN_Agent(env, state_size, action_size,model_path+model_names[0], trained = True)

total_step = 0
all_rewards = []
exploration_rates =[]

for mn in model_names:
    # Reset the environment
    state_stack = []
    dqn.load_my_model(model_path+mn)
    dqn._reset()
    state = env.reset() # start new episode
    action = 1
    state = env.step(action)[0] 
    state = dqn.preprocess_image(state)
    # Need to stack 4 frames together
    state_stack = np.stack([state for _ in range(4)], axis=-1)
    # dqn.see_frames(state_stack)
    done = False
    total_reward = 0
    s = 0

    # Run the game loop without exploration
    #while not done:
    while not done:
        if action == 0 or total_step%100 == 0:
            action = 1
        else:
            action = dqn.get_action(state_stack, explore=False)

        # Take a step in the environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # if total_step > 200:
            # plt.imshow(next_state)
            # plt.show()
            # dqn.see_frames(next_state_stack)
            
        # Add new frame, remove oldest one
        next_state = dqn.preprocess_image(next_state)         
        next_state_stack = np.concatenate([state_stack[..., 1:], next_state[..., np.newaxis]], axis=-1)


        # Update the state
        state_stack = next_state_stack
        total_reward += reward
        total_step += 1
        s+=1
        
# avg_reward = sum(all_rewards[-10:]) / len(all_rewards[-10:]) # calculate the average reward over the last 10 episodes
# avg_explore_rate = sum(exploration_rates[-10:]) / len(exploration_rates[-10:]) # calculate the average exploration rate over the last 10 episodes
    print("total_step: {}, score: {}".format(total_step, total_reward))
    
# plt.plot(model_names,all_rewards)
# plt.xlabel('Model Name')
# plt.ylabel('Total Reward')
# plt.show()
env.close()