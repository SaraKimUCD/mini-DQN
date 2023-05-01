import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import datetime
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from collections import deque

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # print("GPU FOUND")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DQN_Agent:
    # Set the hyperparameters
    def __init__(self, env, state_size, action_size, pathfile, trained):
        # Game environment
        self.env = env
        
        # Path for checkpoint
        self.checkpoint_path = pathfile
    
        # Define state and action size
        self.state_size = state_size
        self.action_size = action_size
        self.shape_size = (-1, 84, 84, 4)

        # Define learning rate
        self.learning_rate = 0.0025
        
        # Define epsilon and decay for training rates
        self.epsilon = 1.0
        self.epsilon_decay = 0.0001
        self.epsilon_min = 0.01

        # Define memory buffer to store replay experience of the state,action,reward,next state, terminal
        self.memory_buffer = deque(maxlen=100000)
        self.batch_size = 32

        # define discount factor
        self.gamma = 0.99
        
        # Definie update frequency of Q Network + steps for training
        self.update_frequency = 10000
        self.steps = 0
        
        # Define CNN model - one for main and one for target (uses next_states)
        if trained:
            print("Found ckpt")
            self.load_my_model(pathfile)
        else:
            self.model = self.CNN_model_create(state_size, action_size)
        self.target_model = self.CNN_model_create(state_size, action_size)
        
        self.total_reward = 0.0
        self.avg_q_value = 0
        
        #opt = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.95, epsilon=0.01)
        self.opt = tf.keras.optimizers.Adam(self.learning_rate, clipnorm=1.0)
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Reset environment + reward
    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        
    def save_my_model(self, full_path):
        model_path = full_path + "_agent_model.h5"
        # target_model_path = full_path + "_target_model.h5"
        self.model.save(model_path)
        # self.target_model.save(target_model_path)
        
    def load_my_model(self, full_path):
        model_path = full_path + "_agent_model.h5"
        # target_model_path = full_path + "_target_model.h5"
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()
        # self.target_model = tf.keras.models.load_model(target_model_path)

    # Build CNN for the Deep Q-learning model to take the preprocessed image as input
    def CNN_model_create(self,state_size, action_size):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84,84,4,)))
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(action_size, activation='linear'))

        # Compile model
        model.compile(loss='mse', optimizer =tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate, rho=0.95, epsilon=0.01) )
        #model.summary()
        return model
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def preprocess_image(self,image):
        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[32:195,14:146]
        image = image[::2,::2] 

        # Resize the image to 84 x 84
        image = cv2.resize(image, (84, 84))
        # Normalize the pixel values
        # image = (image - 128) / (128 - 1)
        image = image.astype(np.uint8)
        return image
        
    def see_frames(self, frames):
        fig, axs = plt.subplots(2, 2)
        for i in range(4):
            row = i // 2
            col = i % 2
            axs[row, col].imshow(frames[:, :, i])
            axs[row, col].set_title('Frame {}'.format(i))
        plt.show()

    # choose an action based on epsilon greedy policy
    def get_action(self, state, explore):
        if np.random.rand() < self.epsilon and explore:
            action = self.env.action_space.sample()
            # print("random:", action)
        else:
            state = np.array(state).reshape(self.shape_size)
            #state = state / 255.0
            q_values = self.model.predict(state,verbose=0)
            action = np.argmax(q_values[0])
            # print("predicted:", action)
        return action  

    # Save the game play as experience
    def store_experience(self, state, action, reward, next_state, done):
        self.memory_buffer.append((state, action, reward, next_state, done))
        return 
    
    # Agent that plays random moves
    def random_agent(self, env):
        total_reward = 0
        total_step = 0
        self.env.reset() # start new episode
        is_done = False
        while not is_done:
            new_action = self.env.action_space.sample() # random action
            next_state, reward, terminated, truncated , info = self.env.step(new_action)
            is_done = terminated or truncated
            total_reward += reward
            total_step += 1
        # Print the total reward for the episode
        print('Random Agent Total Reward: {} in {} steps'.format(total_reward,total_step))
        return

        
    # Function to train our cnn model - it will take batches of experience from the memory_buffer
    def train_experience(self, full_path):
        # First generate training data to store in the memory_buffer as experience so it can be used to sample from to update the Q-model
        if len(self.memory_buffer) < self.batch_size or len(self.memory_buffer) < 50000:
            return

        # Sample a mini-batch of experience from the memory buffer
        minibatch = random.sample(self.memory_buffer, self.batch_size)

        # Preprocess the minibatch and create the necessary arrays
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states).reshape(self.shape_size)
        next_states = np.array(next_states).reshape(self.shape_size)

        states = states / 255.0
        next_states = next_states / 255.0

        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Predict the Q-values for the next state batch using the target model
        target_q = self.target_model.predict(next_states, verbose=0)

        # Compute the target Q-values using Bellman equation
        target_q_values = np.zeros((self.batch_size, self.action_size))
        # target_q_values = self.model.predict(states)
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_q[i])

        # self.model.fit(states, target_q_values, batch_size=self.batch_size, verbose=0)

        # Calculate loss between updated Q-values and target Q-values
        target_q_val = np.array([each[0] for each in target_q_values])
        masks = tf.one_hot(actions, self.action_size)
        with tf.GradientTape() as tape:
            # Predict Q-values for the current states using the main model
            q_values = self.model(states)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.huber(target_q_val, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Decay epsilon with epsilon decay step
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - (self.epsilon_decay/1000)
        
        # Update the target model, save the model and print out training every update_frequency steps
        if self.steps % self.update_frequency == 0:
            self.update_target_model()
        if self.steps % 1000 == 0:
            print("steps:", self.steps, "epsilon:", self.epsilon, "Max Q:", np.mean(max_target_q))

        # Increment the number of steps
        self.steps += 1
        return np.mean(max_target_q)