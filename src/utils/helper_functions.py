# General functions that don't exactly fit anywhere else.
import numpy as np

def get_key_with_most_vals(in_dict):
    maxcount = max(len(v) for v in in_dict.values())
    return [k for k, v in in_dict.items() if len(v) == maxcount]

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

# If we only have one segmentation id, then we need to make a list and add it to the list.
# All of the ids must be integer. not string
# seg_id is either a list or a single variable
def check_annotation_id(seg_id):
    if(type(seg_id) is list):
        return list
    else:
        if(type(seg_id) is str):
            return [int(seg_id)]
        else:
            return seg_id

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

# The annotations in coco are dictionaries with one of the keys being 'segmentation'.
# So, this function places the given list into a dictionary under the 'segmentation' key.
def get_annotation_from_polygon(polygon_list):

    polygon = check_segmentation_polygon(polygon_list)

    return {'segmentation': check_segmentation_polygon(polygon), 'image_id' : 000}

# The annotations in mscoco are stored as a dictionary, under the key 'segmentation'. The value is a [[]].
# So we need to make sure that polygon used with 'segmentation' is in this format
def check_segmentation_polygon(polygon):

    # If the polygon is just a list, then put it in a list.
    if(type(polygon[0]) != list):
        return [polygon]
    else:
        return polygon

def get_set_from_json(input_json):
    return_set = set()
    for key in input_json:
        for val in input_json[key]:
            return_set.add(val)

    return return_set
