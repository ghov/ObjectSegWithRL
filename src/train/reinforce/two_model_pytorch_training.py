import json
import numpy as np
import torch
import torch.nn as nn
from ObjectSegWithRL.src.utils.reinforce_helper import get_initial_state, get_np_reward_vector_from_polygon, \
    apply_action_index_to_state
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from ObjectSegWithRL.src.pytorch_custom_dataset.pytorch_dataset import GregDataset
from ObjectSegWithRL.src.utils.resize_functions import get_coco_instance

from ObjectSegWithRL.src.models.reinforce.greg_cnn import GregNet

from ObjectSegWithRL.src.models.cnn.vgg_utils import vgg19_bn
from ObjectSegWithRL.src.models.reinforce.reward_estimator import RewardEstimator

# The file path for the configuration file
config_file_path = "/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/src/config/config.json"

# Read the config file
with open(config_file_path, 'r') as read_file:
    config_json = json.load(read_file)

# Check which cnn model to use based on config
if config_json["model"] == "vgg19_bn":
    cnn_model = vgg19_bn()
elif config_json["model"] == "gregnet":
    cnn_model = GregNet(config_json['number_of_actions'], config_json['polygon_state_length'])

# Load the state dict for the cnn model.
cnn_model.load_state_dict(torch.load(config_json['file_paths']['model_dir'] + config_json['file_paths']['model_name']))

# Instantiate the reward_estimator model
reward_estimator_model = RewardEstimator(config_json['number_of_actions'], config_json['polygon_state_length'])

test_transformations = transforms.Compose([
    transforms.ToTensor()
])

# Load the optimizer.
optimizer = Adam(reward_estimator_model.parameters(), lr=config_json['learning_rate'],
                 weight_decay=config_json['weight_decay'])

#loss_fn = nn.MSELoss().cuda()
loss_fn = nn.L1Loss().cuda()

annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp1.json'
root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_1/'

dataset = GregDataset(annotation_file_path, root_dir_path, transform=test_transformations)

my_dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

coco = get_coco_instance()

def train(num_epochs):
    best_acc = 0.0
    cuda_avail = torch.cuda.is_available()

    train_loss = 0.0

    if cuda_avail:
        cnn_model.cuda()
        reward_estimator_model.cuda()

    # Set the cnn model to eval()
    cnn_model.eval()

    # Get the initial state of the polygon
    initial_state = get_initial_state(config_json['height_initial'], config_json['width_initial'])

    for epoch in range(num_epochs):
        reward_estimator_model.train()
        train_acc = 0.0
        for i, (images, labels) in enumerate(my_dataloader):

            if cuda_avail:
                images = torch.Tensor.cuda((images.cuda())).view(1, 3, 224, 224).float()

            # Get the image features from the cnn
            image_features = cnn_model.features(images)
            image_features.detach_()

            # Need to continue training on one image until either a stop action is chosen or we have exceeded the
            # maximum number of steps allowed.
            stop_action = False
            step_counter = 0
            previous_state = None

            while((not stop_action) and (step_counter < config_json['max_steps'])):
                #print("The current step is: " + str(step_counter))

                if(step_counter == 0):
                    previous_state = initial_state

                # Clear all accumulated gradients
                optimizer.zero_grad()
                # print(images.shape)

                # convert the previous state to a tensor
                #print(previous_state)
                previous_state_tensor = torch.Tensor.cuda(torch.from_numpy(np.asarray(previous_state))).float()
                outputs = reward_estimator_model.forward(image_features, previous_state_tensor)

                # Get the predicted action
                _, prediction = torch.max(outputs.data, 1)

                #print(str(labels.numpy().tolist()[0]))

                # Get the label, by taking all actions on previous state
                reward_np = get_np_reward_vector_from_polygon(previous_state,
                                                              config_json['coordinate_action_change_amount'],
                labels.numpy().tolist()[0], config_json['height_initial'], config_json['width_initial'], coco,
                                                              config_json['step_cost'],
                                                              config_json['stop_action_reward'])

                reward_tensor = torch.Tensor.cuda(torch.from_numpy(reward_np)).float()

                #print("The shape of the reward tensor is: " + str(reward_tensor.size()))

                # outputs = model.forward(torch.unsqueeze(images, 0))
                height, _, width = labels.shape
                #print("The predicted reward is: " + str(outputs))
                #print("The actual reward is: " + str(reward_tensor.view(1, 17)))

                loss = loss_fn(outputs, reward_tensor.view(1, 17))
                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                optimizer.step()

                train_loss += loss.cpu().data[0] * images.size(0)
                prediction = int(prediction.data.cpu().numpy()[0])

                if(prediction == 16):
                    stop_action = True

                #print("The prediction is: " + str(prediction))

                train_acc += torch.sum(outputs.data == reward_tensor.view(1,17).data)

                # Append the reinforcement step counter
                step_counter += 1

                # Make the new state the previous state
                previous_state = apply_action_index_to_state(previous_state,
                                                             config_json['coordinate_action_change_amount'],
                                                             prediction, config_json['height_initial'],
                                                             config_json['width_initial'])
                #print("The new state is: " + str(previous_state))

        # Compute the average acc and loss over all of the steps taken in this epoch
        train_acc = train_acc / step_counter
        train_loss = train_loss / step_counter

        print("The current epoch is: " + str(epoch))
        #print(train_acc)
        #print(train_loss)

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, train_loss))

    torch.save(reward_estimator_model.state_dict(),
               '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/models/'
               'reinforcement_learning/two_model_rein_' + config_json["model"] + '_' + loss_fn.__str__() + "_" +
               str(round(train_loss.data.numpy().item(), 5)))

def main():
    train(config_json['epochs'])

if __name__ == "__main__":
    main()
