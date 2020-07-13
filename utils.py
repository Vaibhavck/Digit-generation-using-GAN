#@title Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Descriminator(nn.Module):

    def __init__(self, num_channels, features_d):

        super(Descriminator, self).__init__()

        self.descr = nn.Sequential(

            ### batch_size x num_channels x 64 x 64
            nn.Conv2d(num_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            ### batch_size x feqatures_d x 32 x 32
            nn.Conv2d(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2),

            ### batch_size x feqatures_d*2 x 16 x 16
            nn.Conv2d(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2),

            ### batch_size x feqatures_d*2 x 8 x 8
            nn.Conv2d(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2),

            ### batch_size x features_d*8 x 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),

            ### batch_size x 1 x 1 x 1
            nn.Sigmoid()

        )


    def forward(self, x):

        return self.descr(x)



class Generator(nn.Module):

    def __init__(self, num_channels, features_g, channel_noise):

        super(Generator, self).__init__()

        self.gen = nn.Sequential(

            ### batch_size x channel_noise x 1 x 1
            nn.ConvTranspose2d(channel_noise, features_g*16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features_g*16),
            nn.ReLU(),

            ### batch_size x features_g*8 x 4 x 4
            nn.ConvTranspose2d(features_g*16, features_g*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(),

            ### batch_size x features_g x 8 x 8
            nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(),

            ### batch_size x channel_noise x 16 x 16
            nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(),

            ### batch_size x channel_noise x 32 x 32
            nn.ConvTranspose2d(features_g*2, num_channels, kernel_size=4, stride=2, padding=1),
            
            ### batch_size x channel_noise x 64 x 64
            nn.Tanh(),
            
        )


    def forward(self, x):

        return self.gen(x)