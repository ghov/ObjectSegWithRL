# Develop functions that are useful in resizing the image and the corresponding polygon.
# Assume that we are using data in same format as MSCOCO

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np




# Crop the image based on given bounding box and also adjust the dimensions of the polygon
def crop_and_adjust_poly(image_file_path, bbox, polygon):

    read_img = io.imread(image_file_path)



def adjust_poly(bbox, polygon):
    poly_len = len(polygon)
    new_poly = list()

    xmin = bbox[0]
    ymin = bbox[1]

    for i in range(0, poly_len, 2):
        new_poly.append(polygon[i] - xmin)
        new_poly.append(polygon[i+1] - ymin)

    return new_poly

def get_coord_from_bbox(bbox):
    x1 = int(bbox[0])
    x2 = int(bbox[0] + bbox[2])
    y1 = int(bbox[1])
    y2 = int(bbox[1] + bbox[3])

    return x1, x2, y1, y2
