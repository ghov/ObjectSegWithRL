# Functions to help with the rnn/reinforcement learning

import cv2
import numpy as np
from ObjectSegWithRL.src.utils.coco_helper_functions import get_RLE_iou, convert_polygon_to_compressed_RLE, \
    get_coco_instance

from ObjectSegWithRL.src.utils.helper_functions import get_annotation_from_polygon, get_height_width

###
# Constants
reward_multiplier = 100
step_cost = -0.005
coordinate_action_change_amount = 10
###

# Takes an image, a polygon as a list.
# Turns the polygon into a segmentation image.
# Applies bitwise operation on the image and the mask with opencv2.
# Returns the new image as an numpy array
def apply_polygon_to_image(img_np_arr, polygon_list, operation, coco_instance):

    # Get the polygon list as a dictionary in 'segmentation keyword'
    polygon = get_annotation_from_polygon(polygon_list)

    height, width = get_height_width(img_np_arr)

    # Get the mask image from the list
    mask = coco_instance.annToMask_hw(polygon, height, width)

    # Get the correct operation based on input
    if(operation == 'or'):
        bitwise = cv2.bitwise_or
    elif(operation == 'and'):
        bitwise = cv2.bitwise_and
    elif(operation == 'not'):
        bitwise = cv2.bitwise_not

    # Apply mask to image
    new_image = bitwise(img_np_arr, img_np_arr, mask=mask)

    return new_image

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
def get_changed_polygons_from_polygon(previous_polygon, change_amount, height, width):

    # instantiate the new list of polygons
    new_polygon_list = list()

    # Go through each element of the polygon list and change it by the amount.
    # Each change should result in a new polygon.
    new_len = len(previous_polygon) * 2

    for action_index in range(0, new_len):
        new_polygon_list.append(apply_action_index_to_state(previous_polygon, change_amount, action_index, height, width))

    return new_polygon_list

# For each RLE in RLE_list, get the IoU of it and RLE_comparison.
# Return a list of IoUs
def get_RLE_iou_list(RLE_list, RLE_comparison):

    iou_list = list()

    for rle in RLE_list:
        iou_list.append(get_RLE_iou(rle, RLE_comparison))

    return iou_list

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

# After the agent chooses an index action, we apply it to the state
# each point of the vertex has two actions
### We need to check to see if this action will place a negative value in the index.
### If so, then we don't apply that action to the index.
def apply_action_index_to_state(state_polygon, change_amount, action_index, height, width):

    if(len(state_polygon) == action_index/2):
        return state_polygon

    new_state_poly = state_polygon.copy()

    # If the index is even, then we add to that index/2.
    if((action_index % 2) == 0):
        # Check if the action would create a value that is larger than the maximum allowed(height or width)
        if((state_polygon[int(action_index/2)] + change_amount) > height):
            return state_polygon
        else:
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
    new_polygons = get_changed_polygons_from_polygon(polygon, change_amount, height, width)

    # Get the index of the polygons that have negative values
    negative_set = set()
    for index, poly in enumerate(new_polygons):
        if(poly == polygon):
            negative_set.add(index)

    # Convert the new_polygons to RLE format.
    rle_polygons = convert_polygon_to_compressed_RLE(coco_instance, new_polygons, height, width, multiple=True)

    # Compute the IoU for the rle_polygons
    iou_list = get_RLE_iou_list(rle_polygons, ground_truth_rle)

    # Compute the rewards for the new rles
    reward_list = get_reward_list_from_iou_list(100, original_iou, iou_list)

    # Adjust for the negative_indexes
    for val in negative_set:
        reward_list[val] = -1

    # Adjust for the step cost if needed
    if(step_cost != None):
        for i in range(0, len(reward_list)):
            reward_list[i] += step_cost

    # Add the stop action reward if needed
    if(stop_reward != None):
        reward_list.append(stop_reward)

    return np.array(reward_list)

def main():
    ground_truth = [224,105.453,183.979,83.428,176.144, 49.163,136.382,0.193,80.942,0.193,26.063,61.402,5.902, 96.891,
                    0.302,111.581,27.177,191.138,47.339,224,64.144,198.491,72.54,121.368,73.661,104.244,155.422,116.483,
                    215.9,115.257]
    a = [0, 0, 224, 0, 224, 224, 0, 224]
    #a = [15, 60, 94, 0, 209, 69, 15, 199]
    b = get_changed_polygons_from_polygon(a, 5, 224, 224)
    print(np.asarray(b))
    coco = get_coco_instance()

    #print(check_segmentation_polygon(a))
    #print(convert_polygon_to_compressed_RLE(coco, a, 224, 224))
    a_rle = convert_polygon_to_compressed_RLE(coco, a, 224, 224)
    ground_truth_rle = convert_polygon_to_compressed_RLE(coco, ground_truth, 224, 224)
    #print(convert_polygon_to_compressed_RLE(coco, b, 224, 224, multiple=True))
    rles = convert_polygon_to_compressed_RLE(coco, b, 224, 224, multiple=True)
    print(get_RLE_iou(a_rle, ground_truth_rle))
    state_iou = get_RLE_iou(a_rle, ground_truth_rle)
    print(get_RLE_iou_list(rles, ground_truth_rle))
    iou_list = get_RLE_iou_list(rles, ground_truth_rle)
    print(get_reward_list_from_iou_list(100, state_iou, iou_list))

    print(get_np_reward_vector_from_polygon(a, 1, ground_truth, 224, 224, coco, step_cost, 0.001))

    print(apply_action_index_to_state(a, 5, 7, 224, 224))

    img_path = '/media/greghovhannisyan/BackupData1/mscoco/images/train2017/000000172310.jpg'
    img = cv2.imread(img_path)
    res = apply_polygon_to_image(img, a, 'and', coco)

    cv2.imshow('img', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()