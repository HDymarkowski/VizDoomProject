import os
from random import choice
from time import sleep
import vizdoom as vzd
import torch
import json
from torch import nn, optim, distributions
import torch.nn.functional as F


# Neural Netowork module
# In pytorch all neural networks have to derive from "nn.Module"
class CNN(nn.Module):
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

