# Takes as input a 3 channel tensor and returns a vector
import torch.nn as nn
from torch import cat

class GregNet(nn.Module):

    def __init__(self, number_of_actions, len_of_previous_state_vector):
        super(GregNet, self).__init__()

        # This computes the cnn section. After this, we need to resize the features into a vector.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # This take the vector form of the features and computes the linear layers.
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear((256 * 6 * 6) + len_of_previous_state_vector, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
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
            #nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, number_of_actions),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x, previous_state):
        x = self.features(x)
        #print(x.size())
        x = x.view(x.size(0), 256 * 6 * 6)
        #print(x.size())
        #print(previous_state.size())

        x = self.classifier(cat((x, previous_state.view(1,16)), 1))
        #print(x.shape)
        return x
