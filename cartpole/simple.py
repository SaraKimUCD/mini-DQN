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

def Random_games():
    # Each of this episode is its own game.
    for episode in range(10):
        env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            # print(t, next_state, reward, done, info, action)
            if done:
                break
                


env = gym.make("CartPole-v1")
env.reset()
action_size = env.action_space.n
state_size = env.observation_space.shape[0]

print("Action Space {}".format(action_size))
print("State Space {}".format(state_size))

pathfile =  str(pathfile) + r"\model\cart"
dqn = am.DQN_Agent(env, state_size, action_size, pathfile, train = False)

# load model
EPISODES = 200
max_steps = 300
total_step = 0
exp_step = 0       
all_rewards = []
all_Q = []
exploration_rates =[]
all_avg_reward =[]
all_avg_er = []


# First generate training data to store in the memory_buffer as experience so it can be used to sample from to update the Q-model
#dqn.generate_experience()

for episode in range(EPISODES):
    # state_stack = []
    full_path = str(pathfile) + r"\cart_" + str(episode)
    
    state = env.reset() # start new episode
    state = np.reshape(state, [1, state_size])
    #state = dqn.preprocess_image(state)
    # Need to stack 4 frames together
    #state_stack = np.stack([state for _ in range(4)], axis=-1)
    
    # state_stack = deque([state]*4,maxlen=4) # Initialize state stack with 4 identical frames
    done = False
    #total_reward = 0
    i = 0
    while not done:
    # for s in range(max_steps):
        env.render()
        action = dqn.get_action(state, explore=True)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        #next_state = dqn.preprocess_image(next_state)
        if not done or i == max_steps-1:
            reward = reward
        else:
            reward = -100
        # Add new frame, remove oldest one
        #next_state_stack = np.concatenate([state_stack[..., 1:], next_state[..., np.newaxis]], axis=-1)

        total_reward = reward
        #Store the experience
        dqn.store_experience(state, action, reward, next_state, done)

        if done:
            all_rewards.append(total_reward)
            exploration_rates.append(dqn.epsilon)
            avg_reward = sum(all_rewards[-10:]) / len(all_rewards[-10:]) # calculate the average reward over the last 10 episodes
            all_avg_reward.append(avg_reward)
            avg_explore_rate = sum(exploration_rates[-10:]) / len(exploration_rates[-10:]) # calculate the average exploration rate over the last 10 episodes
            all_avg_er.append(avg_explore_rate)
            print("total_step: {}, episode: {}/{}, score: {}, e: {:.2}, avg_reward: {:.2}, avg_explore_rate: {:.2}".format(total_step, episode, EPISODES, total_reward, dqn.epsilon, avg_reward, avg_explore_rate))

        all_Q.append(dqn.train_experience(full_path))
        
        # state_stack = deque(list(state_stack)[1:] + [next_state], maxlen=4) # Update state stack
        i+=1
        total_step += 1
        state = next_state
    
    if episode % 5 == 0:
        dqn.save_my_model(full_path)
        Q_df = pd.DataFrame(all_Q)
        # rewards_df = pd.DataFrame(all_rewards)
        # exp_df = pd.DataFrame(exploration_rates)
        # avgr_df = pd.DataFrame(all_avg_reward)
        # avge_df = pd.DataFrame(all_avg_er)

        Q_df.to_csv(str(pathfile)  +'/all_Q_' + datetime.today().strftime('%Y%m%d') +'.csv', mode='a', header = False, index= False)
        # rewards_df.to_csv(str(pathfile) +'/all_rewards_' + datetime.today().strftime('%Y%m%d') +'.csv', mode='a',header = False, index= False)
        # exp_df.to_csv(str(pathfile) +'/exploration_rates_' + datetime.today().strftime('%Y%m%d') +'.csv', mode='a',header = False, index= False)
        # avgr_df.to_csv(str(pathfile) +'/all_avg_reward_' + datetime.today().strftime('%Y%m%d') +'.csv', mode='a',header = False, index= False)
        # avge_df.to_csv(str(pathfile) +'/all_avg_er_' + datetime.today().strftime('%Y%m%d') +'.csv', mode='a',header = False, index= False)
        
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
env.close()


