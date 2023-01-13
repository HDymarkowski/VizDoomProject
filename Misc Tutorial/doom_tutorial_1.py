# PART 1 : Getting VizDOOM Up and Running

# Import stuff

from vizdoom import *
import random
import time

import numpy as np 

# Setup game

game = vizdoom.DoomGame()
game.load_config('gitDOOM/scenarios/basic.cfg')
game.init()

actions = np.identity(3, dtype=np.uint8) # integer identity matrices for the set of acitons we can take



episodes = 10 # play 10 games
for episode in range(episodes): # loop through episodes
    game.new_episode() # restart the game
    while not game.is_episode_finished(): # while we're not dead and/or the game has not been completed
        state = game.get_state() # the state of the game - all of the iimportant info of the game at that point
        img = state.screen_buffer # The actual image of the game
        info = state.game_variables # game variables is stuff like ammo
        reward = game.make_action(random.choice(actions), 4) # the reward of the action, the second paramater is the frames skipped after every action, due to rewards...
        # ...not always happening immediatly after actions i.e. monster getting shot happens frames after shot fired
        print('reward: ', reward) # reward of individual actions
        time.sleep(0.3)
        

    print('--- RESULT: ', game.get_total_reward()) #the reward of the whole episode
    time.sleep(0.3)

game.close()