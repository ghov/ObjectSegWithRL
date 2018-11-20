import json
import cv2
import numpy as np
from poly_seg_utils.reinforce_helper import get_initial_state, get_np_reward_vector_from_polygon, \
    apply_action_index_to_state, apply_polygon_to_image

from poly_seg_utils.resize_functions import get_coco_instance

# The file path for the configuration file
config_file_path = "/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/src/config/config.json"

# Read the config file
with open(config_file_path, 'r') as read_file:
    config_json = json.load(read_file)

annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp2.json'
image_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_2/8786.jpg'
image_id = '3337'

# Get the coco instance
coco = get_coco_instance()

# Read the image
image = cv2.imread(image_path)

# Read the labels
with open(annotation_file_path, 'r') as ann:
    label = json.load(ann)

label_np = np.asarray(label[image_id]).astype('float')

reward_list = list()

def get_optimal_path(num_epochs):

    # Get the initial state of the polygon
    initial_state = get_initial_state(config_json['height_initial'], config_json['width_initial'])

    # Need to continue training on one image until either a stop action is chosen or we have exceeded the
    # maximum number of steps allowed.
    stop_action = False
    step_counter = 0
    previous_state = None

    # Initiate the list that will hold the new polygon at each step
    step_polygon_list = list()

    # Add the initial state to the step_polygon_list
    step_polygon_list.append(initial_state)

    while ((not stop_action) and (step_counter < config_json['max_steps'])):
        #print("The current step is: " + str(step_counter))

        if(step_counter == 0):
            previous_state = initial_state

        #seg_img = apply_polygon_to_image(np.reshape(image, (224,224,3)), previous_state, 'and', coco)

        #print(label_np.tolist())
        #print(previous_state)

        # Get the label, by taking all actions on previous state
        reward_np = get_np_reward_vector_from_polygon(previous_state,
                                                      config_json['coordinate_action_change_amount'],
        label_np.tolist(), config_json['height_initial'], config_json['width_initial'], coco,
                                                      config_json['step_cost'],
                                                      config_json['stop_action_reward'])

        reward_list.append(reward_np)

        prediction = np.argmax(reward_np)

        #print(str(prediction))
        print(str(reward_np.argsort()[-5:][::-1]))

        if(prediction == 16):
            stop_action = True

        # Append the reinforcement step counter
        step_counter += 1

        # Make the new state the previous state
        previous_state = apply_action_index_to_state(previous_state,
                                                     config_json['coordinate_action_change_amount'],
                                                     prediction, config_json['height_initial'],
                                                     config_json['width_initial'])

        step_polygon_list.append(previous_state)

    return step_polygon_list, reward_list, step_counter

def main():
    polygons, rewards, count = get_optimal_path(config_json['epochs'])
    print(polygons[1:])
    print(len(polygons[1:]))
    print(rewards)
    print(len(rewards))
    print(count)

    write_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp1(copy).json'
    with open(write_file_path, 'w') as write_file:
        json.dump({image_id: polygons[-1]}, write_file)

    avg_rewards = rewards[0]
    for val in rewards:
        avg_rewards += val
    print(avg_rewards/count)

if __name__ == "__main__":
    main()



