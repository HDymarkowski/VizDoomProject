# PART 2 : Converting to a Gym Enviroment

from sre_parse import State
from vizdoom import *
from gym import Env # Import envoriment base class from OpenAI Gym
from gym.spaces import Discrete, Box # Import gym spaces
import cv2 # Import opencv to greyscale stuff
import numpy as np
from matplotlib import pyplot as plt



# Imports for part 3
import torch # pyTorch
import os # for file navigation
from stable_baselines3.common.callbacks import BaseCallback # import callback class from stablebaselines 3
from stable_baselines3.common import env_checker # For checking if enviroment is in correct format

# Imports for Part 4
from stable_baselines3 import PPO # Import PPO for training

# Create VizDOOM OpenAI Gym Enviroment
class VizDoomGym(Env):

    def __init__(self, render = False):


        super().__init__() # Inherit from "Env" class ???
        # TODO: Learn more about OpenAI Gym

        # Set up game
        self.game = vizdoom.DoomGame() # TODO: IDK if "vizdoom." does anything
        self.game.load_config('gitDOOM/scenarios/basic.cfg')

       # Render frame logic
       # Rendering takes a lot of computing and we don't always want to see shit cause we don't care
        if(render == False):
            self.game.set_window_visible(False) # Don't pop up that window - we don't want to see it
        else:
            self.game.set_window_visible(True) # Show us the window

        # Start game after we know whether to render or not
        self.game.init()

        # Set up action space and observation space
        # TODO: I don't really get what these are
        self.observation_space = Box(low = 0, high = 255, shape = (100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3) # 3 actions we can take

 

    def step(self, action): # How we take a step in the enviroment

        # Specify action and take step
        actions = np.identity(3) # 3 actions, represented as [1,0,0], [0,1,0], [0,0,1]
        reward = self.game.make_action(actions[action], 4) # Make the action adn get the reward, 4 = frameskip parameter

        # Get other stuff we need to return
        if (self.game.get_state()):
            state = self.game.get_state().screen_buffer # The next frame of the game
            state = self.greyscale(state) # Does the grayscaling and resizing of the image, implemented in greyscale() method
            ammo = self.game.get_state().game_variables[0]
            info = ammo 
        else: # This logic in case we are finished and there is no next frame - would throw an error otherwise
            # Just returns zeroes for shit
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info":info}

        done = self.game.is_episode_finished() # Whether or not the thing is finished

        return state, reward, done, info

    def render(): # Predifined in Vizdoom but needed to be openAI superclass or smth
        pass

    def reset(self): # What happens when we start a new game
        self.game.new_episode() # Make a new game
        state = self.game.get_state().screen_buffer # Next frame

        return self.greyscale(state) # Return next frame, greyscaled

    def greyscale(self, observation): # Greyscale and resize the game frame, get rid of the bottom bit too
        # Applied in step() and reset()
        # Gets rid of color channel i.e. the 3
        # TODO: Maybe figure out how this works
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY) # Making shit gray - idk how this works

        # Compresses frame down ???
        resize = cv2.resize(gray, (160, 100), interpolation = cv2.INTER_CUBIC) # Reiszes image and scales it down - so we have more pixels to process
        state = np.reshape(resize, (100, 160, 1)) # 

        return state

    def close(self): # Close down the game so it's not floating
        self.game.close()

# Enviroment is now set up

########################################################################################################################################
# TUTORIAL 3

# Setup callback

# Standard training and logging callback
# Used for saving the model in case shit goes wrong
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose = 1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok = True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


# Directories for saving and logging shit
CHECKPOINT_DIR = './train/train_basic' # Checkpoint directory for saving trained reinforcement learning models
LOG_DIR = './logs/log_basic' # 

# Create instance of train and logging callback

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
# check_freq = 10000 means that after every 10000 steps of training our model we're going to save a version of those pyTorch weights for our reinforcement learning agent (can be re-loaded)

#################################################################
# Tutorial 4 - Train the RL Model


env = VizDoomGym(render = True)

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=256)
# CnnPolicy as we are sending in an image
# env passed through
# LOG_DIR used for logging
# Verbose means thaat we're going to have info appearing as we train
# learning_rate can be increased
# n_steps defines batch size for model. 256 = 256 sets of observations, actiosn, log probabililties and values will be stored in the buffer for one iteration
#   > for basic is 300 so we use 256 (don't use whole max for game, just below)


model.learn(total_timesteps=100000, callback=callback )
