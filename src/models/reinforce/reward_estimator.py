import torch.nn as nn
from torch import cat

# Assumes the input is of dimensions (1, 25088)
class RewardEstimator(nn.Module):

    def __init__(self, number_of_actions, len_of_previous_state_vector):
        super(RewardEstimator, self).__init__()

        self.features = nn.Sequential(
            nn.Linear((512 * 7 * 7) + len_of_previous_state_vector, 4096),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, number_of_actions)
        )

    def forward(self, image_features, previous_state):
        x = image_features.view(image_features.size(0), 512 * 7 * 7)
        x = self.features(cat((x, previous_state.view(1, 8)), 1))
        return x
