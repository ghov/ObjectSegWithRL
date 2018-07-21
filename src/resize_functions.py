# Develop functions that are useful in resizing the image and the corresponding polygon.
# Assume that we are using data in same format as MSCOCO
#

import skimage.io as io
import matplotlib.pyplot as plt
from skimage.transform import resize
import json
import numpy as np
import torch
#from ObjectSegWithRL.src.greg_cnn import GregNet
from ObjectSegWithRL.src.greg_cnn_cSigmoid import GregNet

# def import_resize_functions_jup():
#     import sys
#     import os
#     sys.path.insert(0, os.path.abspath('..'))
#
#     from resize_functions import *





# Resize image and filter as a batch.
def resize_img_and_poly_dir(image_directory, polygon_json, filter_set, new_shape, image_write_directory,
                            annotation_write_path):

    new_poly_dict = dict()

    for seg_id in filter_set:
        temp_path = image_directory + str(seg_id) + '.jpg'
        temp_img, temp_poly = resize_img_and_poly(temp_path, new_shape, polygon_json[str(seg_id)])
        new_poly_dict[seg_id] = temp_poly
        io.imsave(image_write_directory + str(seg_id) + '.jpg', temp_img)

    with open(annotation_write_path, 'w') as write_file:
        json.dump(new_poly_dict, write_file, indent=4)


# Provide a set of segmentation_ids_list, the coco_instance, new_shape, polygon_json and return a new json with the resized
# polygons
def resize_polygon_batch(coco_instance, segmentation_ids_int_list, new_shape, polygon_json, new_poly_write_path):

    new_poly = dict()

    # Get the coco annotations for the instances
    annotations = coco_instance.loadAnns(segmentation_ids_int_list)

    # Loop through each annotation and get the bounding box
    for ann in annotations:
        height_old = ann['bbox'][3]
        width_old = ann['bbox'][2]

        new_poly[str(ann['id'])] = resize_polygon(polygon_json[str(ann['id'])], new_shape, (height_old, width_old))

    with open(new_poly_write_path, 'w') as write_file:
        json.dump(new_poly, write_file, indent=4)

# Need to resize just the polygon
def resize_polygon(polygon, new_shape, old_shape):

    # Get the height and width of the old image shape
    height_old, width_old = old_shape

    # Get the height and width of the new_image shape
    height_new, width_new = new_shape

    # Get the x and y ratios
    x_ratio = width_new / width_old
    y_ratio = height_new / height_old

    # initiate the list for the new polygon
    new_poly = list()

    # Adjust the vertices and add them to the new polygon list
    for i in range(0, len(polygon), 2):
        new_poly.append(polygon[i] * x_ratio)
        new_poly.append(polygon[i + 1] * y_ratio)

    adjust_poly_out_of_bounds(new_poly,width_new, height_new)

    return new_poly

# Resize the image and resize the polygon associated with it
def resize_img_and_poly(image_file_path, new_shape, polygon):
    read_img = io.imread(image_file_path)
    height_old, width_old = get_height_width(read_img)
    #height_new, width_new = new_shape

    #x_ratio = width_new / width_old
    #y_ratio = height_new / height_old

    new_img = resize(read_img, new_shape)

    #new_poly = list()

    #for i in range(0, len(polygon), 2):
    #    new_poly.append(polygon[i] * x_ratio)
    #    new_poly.append(polygon[i + 1] * y_ratio)

    #adjust_poly_out_of_bounds(new_poly, width_new, height_new)

    return new_img, resize_polygon(polygon, new_shape, (height_old, width_old))
    #return new_img, new_poly


def get_set_from_json(input_json):
    return_set = set()
    for key in input_json:
        for val in input_json[key]:
            return_set.add(val)

    return return_set

# Since our model expects images with three channels, we need to resize greyscale images to have three channels
def convert_to_three_channel(image_as_numpy):
    # Check if it is greyscale
    if len(image_as_numpy.shape) == 2:
        # If there are only two dimensions, then it only has height and width
        height, width = image_as_numpy.shape

        # Resize the image to have three channels
        temp_img = np.resize(image_as_numpy, (height, width, 3))

        # Return the  new resized image
        return temp_img
    else:
        # If the image already had three channels, then just return it.
        return image_as_numpy


# function to take an image, produce a predicted polygon with a given model and display the segmentation on that image.
#  Need to convert image to tensor and convert output to a list, convert to proper format for coco, then display.
def show_predicted_segmentation_polygon(image_id, image_directory_path, model_state_path, model_instance, coco_instance):

    # Load the image
    # Form the full image path
    image_path = image_directory_path + str(image_id) + '.jpg'

    # Read the image as a numpy array
    # Check if the image has three channels. If not, resize it for three channels.
    temp_image = convert_to_three_channel(io.imread(image_path))

    # make the model use the gpu
    model_instance.cuda()

    # Load the model
    model_instance.load_state_dict(torch.load(model_state_path))

    # convert the image to a cuda float tensor
    image_cuda = torch.Tensor.cuda(torch.from_numpy(temp_image)).float()

    prediction = model_instance.forward(image_cuda.view((1, 3, 224, 224)))
    #prediction = model_instance.forward(torch.unsqueeze(image_cuda, 0))

    return prediction


def get_key_with_most_vals(in_dict):
    maxcount = max(len(v) for v in in_dict.values())
    return [k for k, v in in_dict.items() if len(v) == maxcount]

# Provide a segmentation id, a image directory and a polygon json file.
# The function will load the image, polygon and display the polygon over the image.
def show_image_mask_by_id(coco_instance, segmentation_id, image_directory_path, polygon_json_file_path):

    # Form the image path
    image_path = image_directory_path + str(segmentation_id) + '.jpg'

    # Load the image
    temp_image = io.imread(image_path)

    # Load the json file
    with open(polygon_json_file_path, 'r') as read_file:
        poly_json = json.load(read_file)

    temp_list = list()
    temp_list.append({'segmentation' : [poly_json[str(segmentation_id)]]})

    show_image_with_mask(coco_instance, temp_image, temp_list)

# Load the image and put the annotation on it. Then display it.
def show_image_with_mask(coco_instance, image_np_arr, annotation):
    # Commented code, just in case, the image is provided as a path and not a np_array
    #image = io.imread(image_np_arr)
    #plt.imshow(image)

    plt.imshow(image_np_arr)
    coco_instance.showAnns(annotation)

# If we only have one segmentation id, then we need to make a list and add it to the list.
# All of the ids must be integer. not string
# seg_id is either a list of a single variable
def check_annotation_id(seg_id):
    if(type(seg_id) is list):
        return list
    else:
        if(type(seg_id) is str):
            return [int(seg_id)]
        else:
            return seg_id

# Given a polygon, we are going to return a new polygon that has the number_of_vertices vertices.
# Need to account for the new length being multiple times the current length.
# In this case, will need to make a new vertex between at least 1 newly created vertex.
# For example, if current_n = 4 and desired_n = 10, then we need to get v1.5 = add_vertex(v1,v2) and add_vertex(v1.5,v2)

def add_vertices_to_polygon(polygon, number_of_vertices):
    # If its already of the proper length, then just return it
    if((len(polygon)/2) == number_of_vertices):
        return polygon
    # If it is not of the desired length
    else:
        return




# Given two vertices in two dimensional space, returns a new vertex that is between the two points and on the same line.
def add_vertex(vertex1, vertex2):
    # Could use distance to calculate the best place to add a new point.

    x1, y1 = vertex1
    x2, y2 = vertex2

    x3 = (x1 + x2)/2
    y3 = (y1 + y2)/2

    return (x3, y3)

# Crop the image based on given bounding box and also adjust the dimensions of the polygon
def crop_and_adjust_poly(image_file_path, bbox, polygon):

    read_img = io.imread(image_file_path)

def get_height_width(sci_img):
    img_shape = sci_img.shape
    height, width = None, None
    if(len(img_shape) == 3):
        height, width, _ = img_shape
    elif(len(img_shape) == 2):
        height, width = img_shape
    else:
        print("error occurred with getting image shape")

    return height, width


# This is something we need to do after cropping an image. Simple subtract the vector from the left edge of the bounding
#  box to the left edge of the original image.
def adjust_poly_crop(bbox, polygon):
    poly_len = len(polygon)
    new_poly = list()

    xmin, xmax, ymin, ymax = get_coord_from_bbox(bbox)

    for i in range(0, poly_len, 2):
        new_poly.append(polygon[i] - xmin)
        new_poly.append(polygon[i+1] - ymin)

    # Check if any of the polygon variables are out of bounds of the bounding box.
    adjust_poly_out_of_bounds(new_poly, xmax, ymax)

    return new_poly

def adjust_poly_out_of_bounds(polygon, xmax, ymax):

    # Check if any of the polygon variables are out of bounds of the bounding box.
    for i in range(0, len(polygon), 2):
        if(polygon[i] > xmax):
            polygon[i] = xmax

        if(polygon[i+1] > ymax):
            polygon[i+1] = ymax
# 0 is the smallest x.
# 1 is the smallest y.
# 2 is the width of the image
# 3 is the height of the image.
def get_coord_from_bbox(bbox):
    x1 = int(bbox[0])
    x2 = int(bbox[0] + bbox[2])
    y1 = int(bbox[1])
    y2 = int(bbox[1] + bbox[3])

    return x1, x2, y1, y2

def cross_check_json(vertex_json_path, shape_json_path):

    # Load the jsons
    with open(vertex_json_path, 'r') as read_shape:
        vertex_json = json.load(read_shape)

    with open(shape_json_path, 'r') as read_vertex:
        shape_json = json.load(read_vertex)

    combined_shape_set = set()
    # combine the shape_json
    for key in shape_json:
        combined_shape_set.update(shape_json[key])

    for key in vertex_json:
        for val in vertex_json[key]:
            if val not in combined_shape_set:
                vertex_json[key].remove(val)

    return vertex_json

def main():
    #bbox_crop_dir = '/media/greghovhannisyan/BackupData1/mscoco/images/train2017_crop_bbox/'
    #sample_ann_id = 1874298
    #new_img, new_poly = resize_img_and_poly()
    #image_id, image_directory_path, model_state_path, model_instance, coco_instance

    image_directory_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/30/'
    model_state_path = '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/models/GregNet_MSELoss()_tensor(2504.4006)_RL_cS'
    model_instance = GregNet(15)
    coco_instance = None

    print(show_predicted_segmentation_polygon(1078619, image_directory_path, model_state_path, model_instance, coco_instance))


if __name__ == "__main__":
    main()
