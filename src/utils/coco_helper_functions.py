from pycocotools.coco import COCO
from pycocotools import mask
from poly_seg_utils.helper_functions import check_segmentation_polygon

# Need a coco instance for lots of stuff, including showing annotation on image.
def get_coco_instance(annotation_file=None):

    # If no path is provided, just use the regular 2017 annotations
    #constant_ann_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/instances/instances_train2017.json'
    if(annotation_file == None):
        return COCO()
        #return COCO(constant_ann_path)

    # If a path is provided, then use that annotation file.
    return COCO(annotation_file)

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

# Need to get a floating point value representation of the IoU between two RLE format objects
def get_RLE_iou(RLE_a, RLE_b):
    return mask.iou([RLE_a], [RLE_b], [False])[0][0]