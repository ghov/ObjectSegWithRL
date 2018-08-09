import json
import matplotlib.pyplot as plt
import skimage.io as io
import torch
from ObjectSegWithRL.src.utils.helper_functions import convert_to_three_channel

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
    #io.imshow(image_np_arr)
    #plt.show()

    plt.imshow(image_np_arr)
    #plt.show()
    coco_instance.showAnns(annotation)

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
    prediction_cpu = torch.Tensor.cpu(prediction)
    prediction_np = prediction_cpu.detach().numpy()
    prediction_list = prediction_np.tolist()

    #print(prediction_list[0])
    temp_list = list()
    temp_list.append({'segmentation' : [prediction_list[0]]})

    #io.imshow(io.imread(image_path))
    #plt.show()
    print(prediction_list)
    show_image_with_mask(coco_instance, temp_image, temp_list)

    #return prediction

# function to take an image, produce a predicted polygon with a given model and display the segmentation on that image.
#  Need to convert image to tensor and convert output to a list, convert to proper format for coco, then display.
def show_reinforce_predicted_segmentation_polygon(image_id, image_directory_path, model_state_path, model_instance,
                                                  config_file_path, coco_instance):
    final_polygon = reinforce_poly_test(image_id, config_file_path)[0]

    temp_list = list()
    temp_list.append({'segmentation': [final_polygon]})

    print(final_polygon)

    # Load the image
    # Form the full image path
    image_path = image_directory_path + str(image_id) + '.jpg'

    # Read the image as a numpy array
    # Check if the image has three channels. If not, resize it for three channels.
    temp_image = convert_to_three_channel(io.imread(image_path))

    show_image_with_mask(coco_instance, temp_image, temp_list)