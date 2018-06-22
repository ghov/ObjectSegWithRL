import torch.nn as nn

# Takes as input a vector of state+image and outputs two vectors.
# One vector defines which action to take
# The other vector defines how much action to take


class MyReinforcementLearningNeuralNetwork(nn.Module):



    def __init__(self):