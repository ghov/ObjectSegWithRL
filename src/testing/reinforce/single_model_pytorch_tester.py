# After having trained a model, we would like to test it on some images.
# The function will return a list of the consecutive polygons, which can later be used to view the segmentation with
#  coco.

# This function should only be called for one image.

import json

import torch.nn as nn
import numpy as np
import skimage.io as io
import torch
from ObjectSegWithRL.src.utils.reinforce_helper import get_initial_state, apply_action_index_to_state, \
    apply_polygon_to_image
#from ObjectSegWithRL.src.models.reinforce.greg_vgg import vgg19
from ObjectSegWithRL.src.models.cnn.vgg_utils import vgg19_bn

from ObjectSegWithRL.src.models.reinforce.greg_cnn import GregNet
from ObjectSegWithRL.src.utils.helper_functions import convert_to_three_channel

from ObjectSegWithRL.src.utils.resize_functions import get_coco_instance

def reinforce_poly_test(image_id, config_file_path):

    # Read the config file
    with open(config_file_path, 'r') as read_file:
        config_json = json.load(read_file)

    # Read the image as a numpy array
    # Check if the image has three channels. If not, resize it for three channels.
    temp_image = convert_to_three_channel(io.imread(config_json['file_paths']['image_dir'] + image_id + '.jpg'))

    # Instantiate the model instance
    # Check which model to use based on config
    if config_json["model"] == "vgg19_bn":
        model = vgg19_bn(num_classes=config_json['number_of_actions'], pretrained=False)
    elif config_json["model"] == "gregnet":
        model = GregNet(config_json['number_of_actions'], config_json['polygon_state_length'])

    # Set the mode to evaluate, so dropout is turned off
    model.eval()

    # make the model use the gpu
    model.cuda()

    # Load the model state
    model.load_state_dict(torch.load(config_json['file_paths']['model_dir'] + config_json['file_paths']['model_name']))

    # convert the image to a cuda float tensor
    #image_cuda = torch.Tensor.cuda(torch.from_numpy(temp_image)).float()

    # Get the initial state of the polygon
    initial_state = get_initial_state(config_json['height_initial'], config_json['width_initial'])

    # Go through the reinforcement loop and save each new polygon
    stop_action = False
    step_counter = 0
    previous_state = initial_state

    # Initiate the list that will hold the new polygon at each step
    step_polygon_list = list()

    # Add the initial state to the step_polygon_list
    step_polygon_list.append(initial_state)

    # Get coco instance
    coco = get_coco_instance()

    # Get the softmax fucntion
    soft = nn.Softmax(dim=1)

    # Start the reinforcement learning loop
    while ((not stop_action) and (step_counter < config_json['max_steps'])):
        print("The current step is: " + str(step_counter))

        seg_img = apply_polygon_to_image(temp_image, previous_state, 'and',
                                         coco)
        seg_img_tensor = torch.Tensor.cuda(torch.from_numpy(seg_img).float()).view(1, 3, 224, 224).float()

        # Make the previous state list into a cuda pytorch tensor
        #previous_state_tensor = torch.Tensor.cuda(torch.from_numpy(np.asarray(previous_state))).float()

        # Run a forward step of the model
        outputs = model.forward(seg_img_tensor)
        output_softmax = soft(outputs.view(1, 17))
        print(output_softmax)

        # Get the predicted action
        _, prediction = torch.max(output_softmax.data, 1)
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




