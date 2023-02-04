import os
from random import choice
from time import sleep
import vizdoom as vzd
import torch
import json
from torch import nn, optim, distributions
from model import CNN
import torch.nn.functional as F
import numpy as np

load = True

"""
Changes made:
    > accumulate_episodes = 16 => accumulate_episodes = 8
    > lr = 0.0001 => lr = 0.003
    > Fixed load/saving
"""

# Hyperparameters

lr = 0.0001
save_path = "basic_model.pt"
accumulate_episodes = 8

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
# sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
sleep_time = 0.0

if __name__ == "__main__":

    ### Setup game
    game = vzd.DoomGame()
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    
    # Rendering options

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

    # Set available buttons
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    # print("Available buttons:", [b.name for b in game.get_available_buttons()])
    no_buttons = len(game.get_available_buttons()) # The number of availabe buttons

    # Set available variables (in the game)
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    print("Available game variables:", [v.name for v in game.get_available_game_variables()])

    game.set_episode_timeout(200)
    game.set_episode_start_time(10)

    game.set_window_visible(False) # Whether the traiing is visible or not - makes it slower

    game.set_living_reward(-1)
    game.set_mode(vzd.Mode.PLAYER)

    game.init() # No "game." changes after this will affect the game

    ### Set up actions
    """ 
   actions = [
        [True, False, False],
        [False, True, False],
        [False, False, True]
    ]
    """
    actions = np.identity(no_buttons)

    ### Pre-loop
    
    
    if load == True:
        model = torch.load(save_path)
    else:
        model = CNN(image_height=120, image_width=160, num_actions=no_buttons) # Set up the CNN
    opt = optim.RMSprop(lr=lr, params=model.parameters()) # The optimizer
    out_f = open('log.json', 'w') # The output log file


    i = 72496 # The iteration as we are running an endless loop (while True)
    # Batch = 9062 (multiply that by 8)

    batch_loss = 0.0 # The loss for the current batch
    batch_reward = 0.0 # The reward for the current batch

    ### Training Loop

    while True:

        game.new_episode() # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.

        action_log_probs = [] # The proabilties for each action (after softmax)
        
        while not game.is_episode_finished():

            ## Get state
            state = game.get_state()

            # State consists of:
            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels
            objects = state.objects
            sectors = state.sectors

            ## Play about with the image to make it process right

            screen_buf_t = torch.from_numpy(screen_buf) / 255
            # [H][W][C]
            screen_buf_t = screen_buf_t.transpose(1, 2)
            screen_buf_t = screen_buf_t.transpose(0, 1)
            # [C][H][W]
            screen_buf_t = screen_buf_t.unsqueeze(0)
            # [N][C][H][W]

            ## Get action

            action_logits = model(screen_buf_t) # Get logits
            action_probs = F.softmax(action_logits) # Softmax logits into probabilities
            # print('action_probs', action_probs)
            m = distributions.Categorical(action_probs) # Create distribution for above probabilities
            action = m.sample() # Sample actions to get action
            log_prob = m.log_prob(action)
            # print('log_prob', log_prob)
            action_log_probs.append(log_prob)
            action = action.item()

            r = game.make_action(actions[action])

            ## Sleep if needed
            if sleep_time > 0:
                sleep(sleep_time)

        ## Check how the episode went
        episode_reward = game.get_total_reward()
        episode_reward = episode_reward/ 100 # Normalize it
        per_timestep_losses = [- log_prob * episode_reward for log_prob in action_log_probs]
        per_timestep_losses_t = torch.stack(per_timestep_losses)
        loss = per_timestep_losses_t.sum()

        loss.backward()

        batch_loss = batch_loss + loss.item()
        batch_reward = batch_reward + game.get_total_reward()

        # To avoid steep drop offs, we make it just happen every "accumulate_steps" steps
        if (i + 1) % accumulate_episodes == 0: # "i + 1" so it doesn't do some dodgy stuff initially
            batch = i // accumulate_episodes
            batch_average_reward = batch_reward / accumulate_episodes
            batch_average_loss = batch_loss / accumulate_episodes
            print(f'Batch {batch}, Reward: {batch_average_reward}%.1f, Loss: {batch_average_loss}%.1f')

            opt.step()
            opt.zero_grad()

            ## Write to output log file

            output_dict = {
                'batch': batch,
                'loss': batch_average_loss,
                'reward': batch_average_reward
            }

            out_f.write(json.dumps(output_dict) + ',\n')
            out_f.flush()
            # Reset loss and reward
            batch_loss = 0.0
            batch_reward = 0.0

        ### Save model every once and a while

        if i % 500 == 0:
            torch.save(model, save_path)
            print("Saved model")
        i += 1

    game.close()
