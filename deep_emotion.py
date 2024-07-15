import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepEmotion(nn.Module):
    def __init__(self):
        """
        DeepEmotion class contains the network architecture.
        """
        super(DeepEmotion, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(10, 10, 3)
        self.conv4 = nn.Conv2d(10, 10, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Define batch normalization layer
        self.norm = nn.BatchNorm2d(10)

        # Define fully connected layers
        self.fc1 = nn.Linear(810, 50)
        self.fc2 = nn.Linear(50, 7)

        # Define layers for spatial transformer network (STN)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Define fully connected layers for STN
        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights and biases for the affine transformation in STN
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        """
        Spatial transformer network forward function.
        """
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, input):
        """
        Forward pass of the network.
        """
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out