#!/usr/bin/env python3

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

import os
from random import choice
from time import sleep
import vizdoom as vzd
import torch
import torch.nn as nn
from torch import optim, distributions
import torch.nn.functional as F
import json

# Create a neural network

# Neural Netowork module
# In pytorch all neural networks have to derive from "nn.Module"
class Net(nn.Module):
    # At it's most basic level, the input to this is the image on the screen and the ouput is the action that is to be taken (of the 3 possibl actions)
    # We are only taking one image as input but we could totally take like the 3 previous images as input too so that movement could be detected
    """
    -- The Architecture --

    Input is images, what we see when we're playing (the picuter on the monitor)

    Input is, as per image size [120][160][3]

    Want to do a CNN
        [120][160][3]
        > conv2d(kernel_size = 3 (3x3), 16 feature planes (16 channels), padding = 1) Padding = 1 means size does not change, inly number of channels
        [120][160][16]
        > Max pooling (kernel size = 4 (4x4 grids))
        [30][40][16] (does not change feature planes)
        ReLU()
        > conv2d(kernel_size = 3 (3x3), 16 feature planes (16 channels))
        [30][40][16]
        > Max pooling (kernel size = 4)
        [7][10][16] Rounds up
        ReLU()
        > linear(7*10*16, 3) LOOK BELOW
    
        For the output, we want this to choose between different actions
            > 3 actions [left, right, shoot]
        We need to convert [7 rows][10 columns][16 feature length] into just 3 numbers (where each number represnet how likely each action is)

        We do this via linear layer with output 3 with input product of the 3 numbers (7*10*16)

        Threw in a few ReLU layers too after the max pooling
    """

    def __init__(self, num_actions, image_height:int, image_width:int):

        super().__init__() # Calls init construtor of nn.Module, you need to do this

        h = image_height
        w = image_width

        # Where we define the layers in the network
        self.c1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size = 3, padding = 1) # Conv layers change channels
        # We want conv2d, 3d is for videos
        # input channels is number of color channels
        # output channels is number of featured planes we want to output

        self.pool1 = nn.MaxPool2d(kernel_size = 4) # Max pooling makes image smaller
        
        # After pooling, height and width / 4 (// means round down)
        h //= 4
        w //= 4

        self.c2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size = 3, padding = 1) 

        self.pool2 = nn.MaxPool2d(kernel_size=4)

        h //= 4
        w //= 4

        self.output = nn.Linear(h*w*16, num_actions) # Linear(input, output)
        # 16 = num channels
        # Output is the number of actions
        # Input is size of image at this point


    def forward(self, x):
        # Where we take the input and pass it through the layers

        # This is generalised, in this case batch size will always be one as only one game is running at once
        batch_size = x.size(0)

        x = self.c1(x)
        x = self.pool1(x)
        x = F.relu(x) # Relu also put here

        x = self.c2(x)
        x = self.pool2(x)
        x = F.relu(x) # ... and also here

        # At this point it's [c][h][w]
        # We want it to be [c * h * w] i.e. flatten it out

        x = x.view(batch_size, -1) # Flattens it out into single vector
        x = self.output(x)

        return x


model = Net(num_actions = 3, image_height=120, image_width=160) # len(actions) = 3

LEARNING_RATE = 0.001 # Learning rate
opt = optim.RMSprop(lr = LEARNING_RATE, params=model.parameters()) # Optimizer - could maybe also use Adam? 


######

if __name__ == "__main__":
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()

    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    # game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120) # Changed res to make it really small

    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(True)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed to use.
    # This can be done by adding buttons one by one:
    # game.clear_available_buttons()
    # game.add_available_button(vzd.Button.MOVE_LEFT)
    # game.add_available_button(vzd.Button.MOVE_RIGHT)
    # game.add_available_button(vzd.Button.ATTACK)
    # Or by setting them all at once:
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    # Buttons that will be used can be also checked by:
    print("Available buttons:", [b.name for b in game.get_available_buttons()])

    # Adds game variables that will be included in state.
    # Similarly to buttons, they can be added one by one:
    # game.clear_available_game_variables()
    # game.add_available_game_variable(vzd.GameVariable.AMMO2)
    # Or:
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    print("Available game variables:", [v.name for v in game.get_available_game_variables()])

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(200)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    game.set_window_visible(False)

    # Turns on the sound. (turned off by default)
    # game.set_sound_enabled(True)
    # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented,
    # the sound is only useful for humans watching the game.

    # Sets the living reward (for each move) to -1
    game.set_living_reward(-1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    # Enables engine output to console, in case of a problem this might provide additional information.
    #game.set_console_enabled(True)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    # Define some actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK
    # game.get_available_buttons_size() can be used to check the number of available buttons.
    # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
    actions = [[True, False, False], [False, True, False], [False, False, True]]

    # Run this many episodes
    episodes = 500

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    #sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
    sleep_time = 0

    out_f = open('log.txt')

    for i in range(episodes):
        #print("Episode #" + str(i + 1))

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()
        action_log_probs = []

        while not game.is_episode_finished():

            # Gets the state
            state = game.get_state()

            # Which consists of:
            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer # it's the array size x 3 (3 color channels i.e. 3 values between 0 and 256)
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels
            objects = state.objects
            sectors = state.sectors

            screen_buf_torch = torch.from_numpy(screen_buf) / 255 # Converts it into pytorch tensor with values between 0 and 1 (as opposed to how it usually is, betwene 0 and 256)

            # Currently, this is [H][W][C]
            # We want [C][H][W]
            screen_buf_torch = screen_buf_torch.transpose(1, 2)
            screen_buf_torch = screen_buf_torch.transpose(0, 1)
            # Now it is [C][H][W]

            # We actually need it to be [N][C][H][W]
            # So we need ot put N (the number of batches, in this case 1) at the beginning
            screen_buf_torch = screen_buf_torch.unsqueeze(0) # Adds another dimension with length 1
            # Now we have [N][C][H][W]

            action_logits = model(screen_buf_torch)

            ### Make action logits into probability distribution
            action_probs = F.softmax(action_logits) # Guess softmax makes it probabilities - add up to 1 and are between 0 and 1 inclusive
            
            ### The code for sampling
            m = distributions.Categorical(action_probs)
            action = m.sample() # It's a tensor
            

            log_prob = m.log_prob(action)
            action_log_probs.append(log_prob) # These are used in stead of categorical() (which you can't back propagate through)
            # ... It's a way of doing backpropogation in the presence of discrete sampling

            ###

            #print('Action: ', action.item())
            step_reward = game.make_action(actions[action]) # r is actually reward value

            #print(f'Reward: {step_reward}')

            if sleep_time > 0:
                sleep(sleep_time)


        episode_reward = game.get_total_reward() # Reward for the whole spiode
        # We want to normalize this reward to be no greater than 1 (originally between like - something and 100)
        episode_reward - episode_reward / 100 # Careful with /=, doesn't like it with torch sometimes

        ### Here we calculate loss per timestep
        per_timestep_loss = [-log_prob * episode_reward for log_prob in action_log_probs]
        per_timestep_losses_t = torch.stack(per_timestep_loss) # ?
        loss = per_timestep_losses_t.sum()
        #print(f'Loss: {loss.item()}':.4f)

        print("Episode ", (i + 1), " total reward =  ", episode_reward, " Loss = ", loss.item())


        ### Now we need to do the learning - backpropigaet the loss via an optimizer and stuff

        opt.zero_grad()
        loss.backward()
        opt.step()

        ### Record this an put it in a json file

        out_f.write(json.dumps({ 
            'episode' : i,
            'loss' : loss.item(),
            'reward' : game.get_total_reward()

        }) + '\n'
        )
        out_f.flush() # or else it doesn't write very often



    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()


# TODO: Look at Entropy Stabilisation and initialization or add in another conv layer and smaller image