# After the images have been cropped from the original images, we need to resize the



import json

from pycocotools.coco import COCO

from ObjectSegWithRL.src.utils.resize_functions import adjust_poly_crop

read_directory = '/media/greghovhannisyan/BackupData1/mscoco/images/train2017_crop_bbox/'
write_file_path = '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/train_2017_bbox_crop_polygons_adjusted.json'
annFile = '/media/greghovhannisyan/BackupData1/mscoco/annotations/instances/instances_train2017.json'
vertex_json_path = '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/vertex_count_segmentation_ids.json'
bbox_json_path = '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/bbox_crop_shape_gte10k.json'

# initialize COCO api for instance annotations
coco=COCO(annFile)

with open(vertex_json_path, 'r') as read_vertex:
    shape_json = json.load(read_vertex)

combined_shape_list = list()
# combine the shape_json
for key in shape_json:
    combined_shape_list.extend(shape_json[key])

seg_anns = coco.loadAnns(combined_shape_list)

new_poly = dict()
for ann in seg_anns:
    #print(str(ann['id']) + "yes")
    new_poly[ann['id']] = adjust_poly_crop(ann['bbox'], ann['segmentation'][0])

with open(write_file_path, 'w') as write_file:
    json.dump(new_poly, write_file, sort_keys=True, indent=4)






