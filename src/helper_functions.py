# Functions to help with the rnn/reinforcement learning

import numpy as np
from ObjectSegWithRL.src.resize_functions import get_coco_instance, get_height_width, check_segmentation_polygon
# Need to import mask from coco
from ObjectSegWithRL.cocoapi.PythonAPI.pycocotools import mask

###
# Constants
reward_multiplier = 100
step_cost = -0.5
coordinate_action_change_amount = 10
###

# Set the initial state to be the entire image, with the left corner being (x=0, y=0). The format is (x,y)
def get_initial_state(height, width):

    # Instantiate the list
    temp = list()
    temp[0] = 0
    temp[1] = 0
    temp[2] = width
    temp[3] = 0
    temp[4] = width
    temp[5] = height
    temp[6] = 0
    temp[7] = height

    return temp

# given a polygon vector, index, and amount.
# change the value of the x, y scalars at that index by the amount.
def get_new_polygon_vector(old_vector_np, amount, index):

    new_vector = old_vector_np.copy()
    new_vector[index] = new_vector[index] + amount

# Given a polygon vector and a change amount, we need to change every value of each scalar in the vector by that amount
# and return a list of these vectors.
def get_changed_polygons_from_polygons(previous_polygon, change_amount):

    # instantiate the new list of polygons
    new_polygon_list = list()

    # Go through each element of the polygon list and change it by the amount.
    # Each change should result in a new polygon.
    for index, val in enumerate(previous_polygon):
        # Make a copy of the previous_polygon.
        temp_p = previous_polygon.copy()
        temp_p[index] = previous_polygon[index] + change_amount
        new_polygon_list.append(temp_p)

    return new_polygon_list

# Need to turn a polygon into compressed RLE format
def convert_polygon_to_compressed_RLE(coco_instance, polygon_as_list, height, width):

    return coco_instance.annToRLE_hw({'segmentation' : check_segmentation_polygon(polygon_as_list)}, height, width)


# Need to turn a list of polygons into a list of compressed RLE dictionaries

# Need to get a floating point value representation of the IoU between two RLE format objects
def get_RLE_iou(RLE_a, RLE_b):
    return mask.iou(RLE_a, RLE_b, [False])[0][0]


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


# Given the previous IoU and the new IoU, we need to take the difference of the two and return an appropriate reward.
def get_reward_from_iou(reward_multiplier, previous_iou, new_iou):

    # Compute the difference between the new IoU and the old IoU
    iou_dif = new_iou - previous_iou

    reward = reward_multiplier * iou_dif

# Provide a polygon, an amount, and a ground truth polygon
# Convert the ground truth polygon to RLE format
# Compute the IoU of the ground truth polygon
# Generate a list of polygons by applying that amount to each scalar of the polygon
# Convert these new polygons to RLE format
# Compute the IoU of each of these polygons with the ground truth
# Compute the reward for each IoU by comparing it with the IoU of the original polygon.
# Store these values in a numpy ndarray of (1 X length of polygon)
def get_np_reward_vector_from_polygon(polygon, amount, ground_truth_polygon):




def main():
    a = [0, 0, 0, 0, 0, 0]
    b = get_changed_polygons_from_polygons(a, 1)
    print(np.asarray(b))
    coco = get_coco_instance()
    print(check_segmentation_polygon(a))
    print(convert_polygon_to_compressed_RLE(coco, a, 224, 224))


if __name__ == "__main__":
    main()
