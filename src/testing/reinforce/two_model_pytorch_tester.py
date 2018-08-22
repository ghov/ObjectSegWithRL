# After having trained a model, we would like to test it on some images.
# The function will return a list of the consecutive polygons, which can later be used to view the segmentation with
#  coco.

# This function should only be called for one image.

import json
import numpy as np
import skimage.io as io
import torch
from ObjectSegWithRL.src.utils.reinforce_helper import get_initial_state, apply_action_index_to_state
from ObjectSegWithRL.src.models.cnn.vgg_utils import vgg19_bn

from ObjectSegWithRL.src.models.reinforce.greg_cnn import GregNet
from ObjectSegWithRL.src.utils.helper_functions import convert_to_three_channel
from ObjectSegWithRL.src.models.reinforce.reward_estimator import RewardEstimator


def reinforce_poly_test(image_id, config_file_path):

    # Read the config file
    with open(config_file_path, 'r') as read_file:
        config_json = json.load(read_file)

    # Read the image as a numpy array
    # Check if the image has three channels. If not, resize it for three channels.
    temp_image = convert_to_three_channel(io.imread(config_json['file_paths']['image_dir'] + image_id + '.jpg'))

    # Instantiate the model instance
    # Decide which model to use based on config
    # Check which cnn model to use based on config
    if config_json["model"] == "vgg19_bn":
        cnn_model = vgg19_bn()
    elif config_json["model"] == "gregnet":
        cnn_model = GregNet(config_json['number_of_actions'], config_json['polygon_state_length'])

    # Instantiate the reward_estimator model
    reward_estimator_model = RewardEstimator(config_json['number_of_actions'], config_json['polygon_state_length'])

    # Set the mode to evaluate, so dropout is turned off
    cnn_model.eval()
    reward_estimator_model.eval()

    # make the model use the gpu
    cnn_model.cuda()
    reward_estimator_model.cuda()
    # Load the cnn model state
    cnn_model.load_state_dict(torch.load(config_json['file_paths']['model_dir'] + config_json['file_paths']['model_name']))

    # Load the reward_estimator model state
    reward_estimator_model.load_state_dict(torch.load('/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/'
                                                      'ObjectSegWithRL/data/models/reinforcement_learning/'
                                                      'two_model_rein_vgg19_bn_L1Loss()_0.21679'))

    # convert the image to a cuda float tensor
    image_cuda = torch.Tensor.cuda(torch.from_numpy(temp_image)).float()

    # Get the image features from the cnn
    image_features = cnn_model.forward(image_cuda.view(1, 3, 224, 224))
    image_features.detach_()

    # Get the initial state of the polygon
    #initial_state = get_initial_state(config_json['height_initial'], config_json['width_initial'])
    initial_state = get_initial_state(config_json['height_initial'], config_json['width_initial'])

    # Go through the reinforcement loop and save each new polygon
    stop_action = False
    step_counter = 0
    previous_state = initial_state

    # Initiate the list that will hold the new polygon at each step
    step_polygon_list = list()

    # Add the initial state to the step_polygon_list
    step_polygon_list.append(initial_state)

    # Start the reinforcement learning loop
    while ((not stop_action) and (step_counter < config_json['max_steps'])):
        print("The current step is: " + str(step_counter))

        # Make the previous state list into a cuda pytorch tensor
        previous_state_tensor = torch.Tensor.cuda(torch.from_numpy(np.asarray(previous_state))).float()

        # Run a forward step of the model
        outputs = reward_estimator_model.forward(image_features, previous_state_tensor)

        print(outputs)

        # Get the predicted action
        _, prediction = torch.max(outputs.data, 1)
        prediction = int(prediction.data.cpu().numpy()[0])

        if (prediction == 16):
            stop_action = True

        # Append the reinforcement step counter
        step_counter += 1

        # Make the new state the previous state
        previous_state = apply_action_index_to_state(previous_state, config_json['coordinate_action_change_amount'],
                                                     prediction, config_json['height_initial'], config_json['width_initial'])

        step_polygon_list.append(previous_state)

    return step_polygon_list

def main():

    # The file path for the configuration file
    config_file_path = "/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/src/config/config.json"

    # Set the image id. Probably should be a parameter for the function
    image_id = '3337'

    print(reinforce_poly_test(image_id, config_file_path))


if __name__ == "__main__":
    main()




