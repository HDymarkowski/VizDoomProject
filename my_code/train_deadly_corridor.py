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

"""
Changes made:
    > Now save a seperate model every 2000 runs x
        - So we can see the changes being made
    > Add new buttons x
    > Fix error with shape x

    TODO:
    > New plan to train on low difficulty first
        - Increase difficulty 
        - Could potentially do this manually (I THINK I WILL AT FIRST)
    > Add insentive to not take damage
        - Right now only picks up on health
        - add DAMAGE_TAKEN x
        - Make -ve reward for damage_taken

    A big problem:
        > He just kinda cuts about a lot - training will take a very long time
        > We have -1/frame? (or second, idk) living reward
        > Need some kind of insentive for exploration?
        > Or maybe we could takeaway the living reward
            - Add it back later?
        > https://towardsdatascience.com/explained-curiosity-driven-learning-in-rl-exploration-by-random-network-distillation-72b18e69eb1b
            - Seems like PPO could be used here
"""

# Hyperparameters

load = False
lr = 0.0001
save_path = "basic_model.pt" # This is not actually used
accumulate_episodes = 4 # No clule what this should be
difficulty = 1

# New hyperparameters
save_model_steps = 2000 # How many steps we want to take before we save a new model

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
# sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
sleep_time = 0.0


if __name__ == "__main__":

    ### Setup game
    game = vzd.DoomGame()


    ## NEW: Instead of all of the below options we'll just do...
    game.load_config('C:/Users/dell/Desktop/VizDoomProject/ViZDoom/scenarios/deadly_corridor.cfg') # TODO: We are setting this manually, must be a nicer way of doing 
    # Rendering options

    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "deadly_corridor.wad"))
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120) # Mind this is lower than the normal one
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    

    # Set available buttons
    # print("Available buttons:", [b.name for b in game.get_available_buttons()])
    no_buttons = len(game.get_available_buttons()) # The number of availabe buttons
    #print("NO BUTTONS: " + str(no_buttons))

    # Set available variables (in the game)
    #game.set_available_game_variables([vzd.GameVariable.AMMO2])
    print("Available game variables:", [v.name for v in game.get_available_game_variables()])

    game.set_episode_start_time(10)

    game.set_window_visible(False) # Whether the traiing is visible or not - makes it slower

    #game.set_living_reward(-1) # No clue if I should keep this on or not???
    game.set_mode(vzd.Mode.PLAYER)

    game.init() # No "game." changes after this will affect the game

    ### Set up actions
    actions = np.identity(no_buttons)
    #print("Actions = ", actions)

    ### Pre-loop
    
    
    if load == True:
        model = torch.load(save_path) # Load from the save_path
    else:
        model = CNN(image_height=120, image_width=160, num_actions=no_buttons) # Create a new CNN
    opt = optim.RMSprop(lr=lr, params=model.parameters()) # The optimizer
    out_f = open('log.json', 'w') # The output log file


    i = 0 # The iteration as we are running an endless loop (while True)

    batch_loss = 0.0 # The loss for the current batch
    batch_reward = 0.0 # The reward for the current batch

    # Set initial difficulty
    game.set_doom_skill(difficulty)


    ### Training Loop

    while True:

        game.new_episode() # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.

        action_log_probs = [] # The proabilties for each action (after softmax)
 
        prev_health = 100 # DODGY

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
            #print(screen_buf_t.shape)

            ## Get action

            action_logits = model(screen_buf_t)
            action_probs = F.softmax(action_logits) # Get action probabilities
            m = distributions.Categorical(action_probs) # Create distribution for above probabilities
            action = m.sample() # Sample actions to get action
            log_prob = m.log_prob(action)
            action_log_probs.append(log_prob)
            action = action.item()

            r = game.make_action(actions[action])

            health = vars[0]
            damage_reward = 0
            total_damage_taken = 0
            if (health < prev_health):
                damage_taken = prev_health - health
                #print(f'took {damage_taken} damage!')
                # -ve reward or something here
                
                prev_health = health
                total_damage_taken += damage_taken


            ## Sleep if needed
            if sleep_time > 0:
                sleep(sleep_time)

        ## Check how the episode went
        damage_reward = total_damage_taken * 0.2 # "0.2" is a total guess
        episode_reward = game.get_total_reward() - damage_reward
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
            print(f'Batch {batch}, Reward: {batch_average_reward}, Loss: {batch_average_loss}')

            opt.step()
            opt.zero_grad()

            ## Write to output log file

            output_dict = {
                'batch': batch,
                'loss': batch_average_loss,
                'reward': batch_average_reward,
                'difficulty': difficulty
            }

            out_f.write(json.dumps(output_dict) + ',\n')
            out_f.flush()

            ## Difficulty check and change here
            """
            if (good things happening) and difficulty < 5:
                game.set_doom_skill + 1
            """


            # Reset loss and reward
            batch_loss = 0.0
            batch_reward = 0.0

        ### Save model every once and a while

        if i % save_model_steps == 0:
            save_path = "model_" + str(i) + ".pt"
            torch.save(model, save_path)
            print("Saved model")
        i += 1

    game.close()
