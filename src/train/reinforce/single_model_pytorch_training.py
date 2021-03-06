import json

import numpy as np
import torch
import torch.nn as nn
from ObjectSegWithRL.src.utils.reinforce_helper import get_initial_state, get_np_reward_vector_from_polygon, \
    apply_action_index_to_state, apply_polygon_to_image
#from ObjectSegWithRL.src.models.reinforce.greg_vgg import vgg19
from ObjectSegWithRL.src.models.cnn.vgg_utils import vgg19_bn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from ObjectSegWithRL.src.models.reinforce.greg_cnn import GregNet
from ObjectSegWithRL.src.pytorch_custom_dataset.pytorch_dataset import GregDataset
from ObjectSegWithRL.src.utils.resize_functions import get_coco_instance

#torch.manual_seed(3747464832012123654)
#torch.cuda.manual_seed(7885197066917937)

# The file path for the configuration file
config_file_path = "/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/src/config/config.json"

print('the seed is: ' + str(torch.initial_seed()))
print('The cuda seed is: ' + str(torch.cuda.initial_seed()))

# Read the config file
with open(config_file_path, 'r') as read_file:
    config_json = json.load(read_file)


# Check if gpu is available
cuda_avail = torch.cuda.is_available()

# Check which model to use based on config
if config_json["model"] == "vgg19_bn":
    model = vgg19_bn(num_classes = config_json['number_of_actions'], pretrained=False)
elif config_json["model"] == "gregnet":
    model = GregNet(config_json['number_of_actions'], config_json['polygon_state_length'])

if cuda_avail:
    model.cuda()
    #print("yes")

test_transformations = transforms.Compose([
    transforms.ToTensor()
])

optimizer = Adam(model.parameters(), lr=config_json['learning_rate'], weight_decay=config_json['weight_decay'])

#loss_fn = nn.MSELoss().cuda()
loss_fn = nn.L1Loss().cuda()

#annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp2.json'
#root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_1/'

root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_30/'
annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp30.json'

#annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/30_vertex_poly_adjusted.json'
#root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_2/'

dataset = GregDataset(annotation_file_path, root_dir_path, transform=test_transformations)

my_dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

coco = get_coco_instance()

# Instantiate the action for the random choice
actions = np.array([0, 2, 5, 6, 9, 11, 12, 15])

# Get the softmax fucntion
soft = nn.Softmax(dim=1)

# Used to cache the rewards so we don't need to recalculate them again
reward_dict = dict()

# Used to cache the states so we don't need to recalculate them again
state_dict = dict()

# Used to cache the segmented images so we don't need to recalculate them again
seg_img_dict = dict()

def train(num_epochs):
    best_acc = 0.0
    cuda_avail = torch.cuda.is_available()

    train_loss = 0.0

    if cuda_avail:
        model.cuda()
        #print("yes")

    # Get the initial state of the polygon
    initial_state = get_initial_state(config_json['height_initial'], config_json['width_initial'])

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(my_dataloader):

            if epoch == 0:
                state_dict[i] = list()
                reward_dict[i] = list()
                seg_img_dict[i] = list()

            # Need to continue training on one image until either a stop action is chosen or we have exceeded the
            # maximum number of steps allowed.
            stop_action = False
            step_counter = 0
            previous_state = None

            # Loop through the set actions as a starting point
            for action_index, set_action in enumerate(actions):
                train_acc = 0.0
                train_loss = 0.0
                step_counter = 0

                #while((not stop_action) and (step_counter < config_json['max_steps'])):
                while step_counter < config_json['max_steps']:
                    #print("The current step is: " + str(step_counter))

                    if(step_counter == 0):
                        #previous_state = initial_state
                        previous_state = apply_action_index_to_state(initial_state,
                                                                     config_json['coordinate_action_change_amount'],
                                                                     set_action, config_json['height_initial'],
                                                                     config_json['width_initial'])
                        previous_state = apply_action_index_to_state(previous_state,
                                                                     config_json['coordinate_action_change_amount'],
                                                                     set_action, config_json['height_initial'],
                                                                     config_json['width_initial'])

                    # If we are on the first epoch, then calculate and store for future use.
                    if epoch == 0:
                        seg_img = apply_polygon_to_image(np.reshape(images.data.cpu().numpy(), (224,224,3)), previous_state, 'and', coco)
                        seg_img_tensor = torch.Tensor.cuda(torch.from_numpy(seg_img).float()).view(1, 3, 224, 224).float()
                        seg_img_dict[i].append(seg_img_tensor)
                    else:
                    # If we are not in the first epoch, then just get the value from the cache
                        #print(action_index)
                        #print(step_counter)
                        #print((action_index * config_json['max_steps']) + step_counter)
                        #print(len(seg_img_dict[i]))
                        seg_img_tensor = seg_img_dict[i][(action_index * config_json['max_steps']) + step_counter]

                    # Clear all accumulated gradients
                    optimizer.zero_grad()
                    # print(images.shape)

                    # convert the previous state to a tensor
                    #print(previous_state)
                    #previous_state_tensor = torch.Tensor.cuda(torch.from_numpy(np.asarray(previous_state))).float()
                    outputs = model.forward(seg_img_tensor)

                    # Get the predicted action
                    _, prediction = torch.max(outputs.data, 1)

                    #print(str(labels.numpy().tolist()[0]))

                    # Get the label, by taking all actions on previous state
                    if epoch == 0:
                        reward_np = get_np_reward_vector_from_polygon(previous_state,
                                                                      config_json['coordinate_action_change_amount'],
                        labels.numpy().tolist()[0], config_json['height_initial'], config_json['width_initial'], coco,
                                                                      config_json['step_cost'],
                                                                      config_json['stop_action_reward'])

                        reward_tensor = torch.Tensor.cuda(torch.from_numpy(reward_np)).float()

                        reward_tensor_softmax = soft(reward_tensor.view(1, 17))

                        # Cache the result for future use
                        reward_dict[i].append(reward_tensor_softmax)
                    else:
                    # If we are not in the first epoch, then just get the value from the cache
                        reward_tensor_softmax = reward_dict[i][(action_index * config_json['max_steps']) + step_counter]

                    # Convert the output to a softmax
                    output_softmax = soft(outputs.view(1, 17))

                    #print("The shape of the reward tensor is: " + str(reward_tensor.size()))

                    # outputs = model.forward(torch.unsqueeze(images, 0))
                    height, _, width = labels.shape
                    #print("The predicted reward is: " + str(output_softmax))
                    #print("The actual reward is: " + str(reward_tensor.view(1, 17)))
                    #print("The softmax reward is: " + str(reward_tensor_softmax))

                    loss = loss_fn(output_softmax, reward_tensor_softmax)
                    #print('The current loss is: ' + str(loss.cpu().data[0]))
                    # Backpropagate the loss
                    loss.backward()

                    # Adjust parameters according to the computed gradients
                    optimizer.step()

                    train_loss += loss.cpu().data[0] * images.size(0)
                    #prediction = int(prediction.data.cpu().numpy()[0])
                    #prediction = np.random.choice(actions)

                    prediction = np.argmax(reward_np)

                    if(prediction == 16):
                        stop_action = True

                    #print("is prediction 16? " + str(prediction == 16))

                    #print("The prediction is: " + str(prediction))

                    train_acc += torch.sum(outputs.data == reward_tensor.view(1,17).data)

                    # Append the reinforcement step counter
                    step_counter += 1

                    #if (step_counter == 0):
                    #    prediction = set_action

                    # Make the new state the previous state
                    previous_state = apply_action_index_to_state(previous_state,
                                                                 config_json['coordinate_action_change_amount'],
                                                                 prediction, config_json['height_initial'],
                                                                 config_json['width_initial'])


                    #print("The new state is: " + str(previous_state))

                # Compute the average acc and loss over all training images
                train_acc = train_acc / step_counter
                train_loss = train_loss / step_counter

                print("The current epoch is: " + str(epoch))
                #print(train_acc)
                #print(train_loss)

                # Print the metrics
                print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, train_loss))

    torch.save(model.state_dict(),
               '/media/greghovhannisyan/BackupData1/models/'
               'reinforcement_learning/one_model_rein_' + str(torch.initial_seed()) + '_' + str(torch.cuda.initial_seed())
                                                              + config_json["model"] + '_' + loss_fn.__str__() + "_" +
               str(round(train_loss.data.numpy().item(), 5)))

    #torch.save(model.state_dict(),
    #           '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/models/'
    #           'reinforcement_learning/one_model_rein_' + config_json["model"] + '_' + loss_fn.__str__() + "_" +
    #           str(round(train_loss.data.numpy().item(), 5)))

    print('the seed is: ' + str(torch.initial_seed()))
    print('The cuda seed is: ' + str(torch.cuda.initial_seed()))
def main():

    #train(config_json['epochs'])

    for i in range(0, 10):
        train(config_json['epochs'])

if __name__ == "__main__":
    main()



