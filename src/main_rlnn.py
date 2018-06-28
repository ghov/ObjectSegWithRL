import torch.nn as nn

# Takes as input a vector of state+image and outputs two vectors.
# One vector defines which action to take
# The other vector defines how much action to take

# output size variables:
# 1. action vector = number of points + 1
# 2. action amount vector = 2 * number of points
# 3. classification vector = 80 object categories and 91 stuff categories


# with 4 points, the output size will be = 5 + 8 + 80 = 93


# Expecting input of size 1x4096

class GregNet(nn.Module):

    def __init__(self, number_of_actions, number_of_classes):
        super(GregNet, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, number_of_actions + 1 + 2*(number_of_actions) + number_of_classes),
        )

    # Overwrites the forward method of the Module class
    def forward(self, x):
        return self.classifier(x)


