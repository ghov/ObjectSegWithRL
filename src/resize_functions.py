# Develop functions that are useful in resizing the image and the corresponding polygon.
# Assume that we are using data in same format as MSCOCO
#

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import json


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

# Resize the image and resize the polygon associated with it
def resize_img_and_poly(image_file_path, new_shape, polygon):
    read_img = io.imread(image_file_path)
    height_old, width_old = get_height_width(read_img)
    height_new, width_new = new_shape

    x_ratio = width_new/width_old
    y_ratio = height_new/height_old

    new_img = resize(read_img, new_shape)

    new_poly = list()

    for i in range(0, len(polygon), 2):
        new_poly.append(polygon[i] * x_ratio)
        new_poly.append(polygon[i+1] * y_ratio)

    return new_img, new_poly

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

    combined_shape_list = list()
    # combine the shape_json
    for key in shape_json:
        combined_shape_list.extend(shape_json[key])

    for key in vertex_json:
        for val in vertex_json[key]:
            if val not in combined_shape_list:
                vertex_json[key].remove(val)

    return vertex_json

def main():
    bbox_crop_dir = '/media/greghovhannisyan/BackupData1/mscoco/images/train2017_crop_bbox/'
    sample_ann_id = 1874298
    new_img, new_poly = resize_img_and_poly()

if __name__ == "__main__":
    main()
