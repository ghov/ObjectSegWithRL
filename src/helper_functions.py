# Functions to help with the rnn/reinforcement learning

import numpy as np
from ObjectSegWithRL.src.resize_functions import get_coco_instance, get_height_width, check_segmentation_polygon
# Need to import mask from coco
from ObjectSegWithRL.cocoapi.PythonAPI.pycocotools import mask

###
# Constants
reward_multiplier = 100
step_cost = -0.005
coordinate_action_change_amount = 10
###

# After the agent chooses an index action, we apply it to the state
# each point of the vertex has two actions
### We need to check to see if this action will place a negative value in the index.
### If so, then we don't apply that action to the index.
def apply_action_index_to_state(state_polygon, change_amount, action_index):

    if(len(state_polygon) == action_index/2):
        return state_polygon

    new_state_poly = state_polygon.copy()

    # If the index is even, then we add to that index/2.
    if((action_index % 2) == 0):
        new_state_poly[int(action_index/2)] += change_amount
    else:
        # Check if the action would create a negative value in that index.
        if ((state_polygon[int((action_index-1)/2)] - change_amount) < 0):
            return state_polygon
        else:
        # if the index is odd, we subtract from index/2 -1 of that index.
            new_state_poly[int((action_index-1)/2)] -= change_amount

    return new_state_poly

# Set the initial state to be the entire image, with the left corner being (x=0, y=0). The format is (x,y)
# This really depends on the size that we need, so it should be more flexible
def get_initial_state(height, width):

    # Instantiate and populate the list
    temp = list([0, 0, width, 0, width, height, 0, height])
    #temp = list([0, 0, width/2, 0, width, 0, width, height/2, width, height, width/2, height, 0, height, 0, height/2])

    return temp

# given a polygon vector, index, and amount.
# change the value of the x, y scalars at that index by the amount.
def get_new_polygon_vector(old_vector_list, amount, index):

    if(len(old_vector_list) == index):
        return old_vector_list

    new_vector = old_vector_list.copy()
    new_vector[index] = new_vector[index] + amount

    return new_vector

# Given a polygon vector and a change amount, we need to change every value of each scalar in the vector by that amount
# and return a list of these vectors.
# So we should have len(list of poly) * 2 results
def get_changed_polygons_from_polygon(previous_polygon, change_amount):

    # instantiate the new list of polygons
    new_polygon_list = list()

    # Go through each element of the polygon list and change it by the amount.
    # Each change should result in a new polygon.
    new_len = len(previous_polygon)

    for i in range(0, new_len):
        # Make a copy of the original
        temp_p = previous_polygon.copy()

        # Add the amount to the even index
        temp_p[i] += change_amount

        # Add this polygon to the list of polygons
        new_polygon_list.append(temp_p)

        # Make a copy of the original
        temp_p = previous_polygon.copy()

        # Subtract the amount from the add index
        temp_p[i] -= change_amount

        # Add this polygon to the list of polygons
        new_polygon_list.append(temp_p)

    # for index, val in enumerate(previous_polygon):
    #     # Make a copy of the previous_polygon.
    #     temp_p = previous_polygon.copy()
    #     temp_p[index] = previous_polygon[index] + change_amount
    #     new_polygon_list.append(temp_p)

    return new_polygon_list

# Need to turn a polygon into compressed RLE format
def convert_polygon_to_compressed_RLE(coco_instance, polygon_as_list, height, width, multiple=False):

    if not multiple:
        return coco_instance.annToRLE_hw({'segmentation': check_segmentation_polygon(polygon_as_list)}, height, width)
    else:
        # Instantiate a new list to store the results
        new_rle_list = list()

        for poly in polygon_as_list:
            new_rle_list.append(coco_instance.annToRLE_hw({'segmentation': check_segmentation_polygon(poly)}
                                                          , height, width))
        return new_rle_list


# For each RLE in RLE_list, get the IoU of it and RLE_comparison.
# Return a list of IoUs
def get_RLE_iou_list(RLE_list, RLE_comparison):

    iou_list = list()

    for rle in RLE_list:
        iou_list.append(get_RLE_iou(rle, RLE_comparison))

    return iou_list

# Need to get a floating point value representation of the IoU between two RLE format objects
def get_RLE_iou(RLE_a, RLE_b):
    return mask.iou([RLE_a], [RLE_b], [False])[0][0]


# Need to get IoU from two polygons represented as lists
def get_iou_from_polygon(polygon_a_as_list, polygon_b_as_list, coco_instance, height, width):

    #Convert the polygons to RLE format
    rle_a = convert_polygon_to_compressed_RLE(coco_instance, polygon_a_as_list, height, width)
    rle_b = convert_polygon_to_compressed_RLE(coco_instance, polygon_b_as_list, height, width)

    return get_RLE_iou(rle_a, rle_b)


# Compute the iou of two segmentation images provided as numpy arrays of equal size
def get_iou_from_np_segmentation(ground_truth_segmentation_np_image, predicted_segmentation_np_image):

    # Compute the binary intersection
    intersection = np.logical_and(ground_truth_segmentation_np_image, predicted_segmentation_np_image)

    # Compute the binary union
    union = np.logical_or(ground_truth_segmentation_np_image, predicted_segmentation_np_image)

    # Compute the IoU
    iou = np.sum(intersection) / np.sum(union)

# Given a polygon, we need to convert it to a segmentation mask as a numpy array.
def get_segmentation_image_from_polygon(polygon_numpy_vector):
    return None

# For each value in the new_iou_list, get the reward by comparing it with the previous IoU
# return a list of rewards
def get_reward_list_from_iou_list(reward_multiplier, previous_iou, new_iou_list):

    # Instantiate the reward_list
    reward_list = list()

    for iou in new_iou_list:
        reward_list.append(get_reward_from_iou(reward_multiplier, previous_iou, iou))

    return reward_list

# Given the previous IoU and the new IoU, we need to take the difference of the two and return an appropriate reward.
def get_reward_from_iou(reward_multiplier, previous_iou, new_iou):

    # Compute the difference between the new IoU and the old IoU
    iou_dif = new_iou - previous_iou

    reward = reward_multiplier * iou_dif

    return reward

# Provide a polygon, an amount, and a ground truth polygon
# Convert the ground truth polygon to RLE format
# Compute the IoU of the original polygon with the ground truth polygon
# Generate a list of polygons by applying that amount to each scalar of the polygon
# Convert these new polygons to RLE format
# Compute the IoU of each of these polygons with the ground truth
# Compute the reward for each IoU by comparing it with the IoU of the original polygon.
# Store these values in a numpy ndarray of (1 X length of polygon)
def get_np_reward_vector_from_polygon(polygon, change_amount, ground_truth_polygon, height, width, coco_instance,
                                      step_cost = None, stop_reward = None):

    # Convert the ground truth polygon to RLE
    ground_truth_rle = convert_polygon_to_compressed_RLE(coco_instance, ground_truth_polygon, height, width)

    # Convert the original polygon to compressed RLE
    original_poly_rle = convert_polygon_to_compressed_RLE(coco_instance, polygon, height, width)

    # Compute the IoU of the original polygon with the ground truth polygon
    original_iou = get_RLE_iou(original_poly_rle, ground_truth_rle)

    # get_changed_polygons_from_polygons
    new_polygons = get_changed_polygons_from_polygon(polygon, change_amount)

    # Get the index of the polygons that have negative values
    negative_set = set()
    for index, poly in enumerate(new_polygons):
        for val in poly:
            if(val < 0):
                negative_set.add(index)

    # Convert the new_polygons to RLE format.
    rle_polygons = convert_polygon_to_compressed_RLE(coco_instance, new_polygons, height, width, multiple=True)

    # Compute the IoU for the rle_polygons
    iou_list = get_RLE_iou_list(rle_polygons, ground_truth_rle)

    # Compute the rewards for the new rles
    reward_list = get_reward_list_from_iou_list(100, original_iou, iou_list)

    # Adjust for the negative_indexes
    for val in negative_set:
        reward_list[val] = -100

    # Adjust for the step cost if needed
    if(step_cost != None):
        for i in range(0, len(reward_list)):
            reward_list[i] += step_cost

    # Add the stop action reward if needed
    if(stop_reward != None):
        reward_list.append(stop_reward)

    return np.array(reward_list)

def main():
    a = [0, 0, 0, 0, 0, 0]
    b = get_changed_polygons_from_polygon(a, 1)
    print(np.asarray(b))
    coco = get_coco_instance()

    print(check_segmentation_polygon(a))
    print(convert_polygon_to_compressed_RLE(coco, a, 224, 224))
    a_rle = convert_polygon_to_compressed_RLE(coco, a, 224, 224)
    print(convert_polygon_to_compressed_RLE(coco, b, 224, 224, multiple=True))
    rles = convert_polygon_to_compressed_RLE(coco, b, 224, 224, multiple=True)
    print(get_RLE_iou(a_rle, a_rle))
    print(get_RLE_iou_list(rles, a_rle))
    iou_list = get_RLE_iou_list(rles, a_rle)
    print(get_reward_list_from_iou_list(100, 1.0, iou_list))

    print(get_np_reward_vector_from_polygon(a, 1, a, 224, 224, coco, 0.5, 0.001))

    print(apply_action_index_to_state(a, 10, 11))

if __name__ == "__main__":
    main()
