import torch.nn as nn

class RewardEstimator(nn.Module):

    def __init__(self, image_features, number_of_actions, len_of_previous_state_vector, init_weights=True):
        super(RewardEstimator, self).__init__()



