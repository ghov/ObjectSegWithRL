import torch
from ObjectSegWithRL.src.greg_cnn import GregNet

model_path = '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/models/GregNet_tensor(140.8939)'

model = GregNet(15)
model.load_state_dict(torch.load(model_path))




