import os
import vizdoom as vzd
import torch
import json
from torch import nn, optim, distributions
from model import CNN
import torch.nn.functional as F
import argparse
from time import sleep

def run(args):

    # Make the game
    game = vzd.DoomGame()


    # Set up scenario and map
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
    game.set_doom_map("map01")

    ### Configuration settings
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    # Bunch of visual stuff disabled cause not needed and complicates image
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False) 
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)  
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True) 

    # Set available buttons
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])

    # Episode Configurations
    game.set_episode_timeout(200)
    game.set_episode_start_time(10)
    ##
    game.set_window_visible(True)
    ##
    game.set_living_reward(-1)
    game.set_mode(vzd.Mode.PLAYER)

    game.init()

    # Define some actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK
    # game.get_available_buttons_size() can be used to check the number of available buttons.
    # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
    actions = [
        [True, False, False],
        [False, True, False],
        [False, False, True]
    ]

    # The pause after each action
    sleep_time = 0.05

    # The model, loaded from the argument provided
    model = torch.load(args.model_name)

    ## The running loop

    # Can change to a conditional
    while True:
        # Every iteration is a new episode

        game.new_episode()

        while not game.is_episode_finished():
            state = game.get_state() # Game state
            screen_buf = state.screen_buffer # The actual image on screen

            ### Get action from screen buffer
            screen_buf_t = torch.from_numpy(screen_buf) / 255
            # [H][W][C]
            screen_buf_t = screen_buf_t.transpose(1, 2)
            screen_buf_t = screen_buf_t.transpose(0, 1)
            # [C][H][W]
            screen_buf_t = screen_buf_t.unsqueeze(0)
            # [N][C][H][W]
            action_logits = model(screen_buf_t)
            action_probs = F.softmax(action_logits)

            m = distributions.Categorical(action_probs)
            action = m.sample()
            action = action.item()

            # Make action
            r = game.make_action(actions[action])

            if sleep_time > 0:
                sleep(sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    args = parser.parse_args()
    run(args)