from torch.optim import Adam
import torch.nn as nn
import torch
from ObjectSegWithRL.src.reinforce.greg_cnn import GregNet
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from ObjectSegWithRL.src.helper_functions import get_initial_state, get_np_reward_vector_from_polygon,\
    get_new_polygon_vector, apply_action_index_to_state
from ObjectSegWithRL.src.pytorch_dataset import GregDataset
from ObjectSegWithRL.src.resize_functions import get_coco_instance

reward_multiplier = 100
step_cost = -0.005
coordinate_action_change_amount = 5
number_of_actions = 17
polygon_state_length = 8
height_initial = 224
width_initial = 224
max_steps = 10000
stop_action_reward = 0
#stop_action_reward = 0.00001
#height, width = (224, 224)


cuda_avail = torch.cuda.is_available()

model = GregNet(number_of_actions, polygon_state_length)

if cuda_avail:
    model.cuda()
    #print("yes")

test_transformations = transforms.Compose([
    transforms.ToTensor()
])
# test_transformations = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#
# ])

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

#loss_fn = nn.MSELoss().cuda()
loss_fn = nn.L1Loss().cuda()

annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp2.json'
root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_2/'

#annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/30_vertex_poly_adjusted.json'
#root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/30/'

dataset = GregDataset(annotation_file_path, root_dir_path, transform=test_transformations)

my_dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

coco = get_coco_instance()

def train(num_epochs):
    best_acc = 0.0
    cuda_avail = torch.cuda.is_available()

    train_loss = 0.0

    if cuda_avail:
        model.cuda()
        #print("yes")

    # Get the initial state of the polygon
    initial_state = get_initial_state(height_initial, width_initial)

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        for i, (images, labels) in enumerate(my_dataloader):


            if cuda_avail:

                images = torch.Tensor.cuda((images.cuda())).view(1, 3, 224, 224).float()
                #print(images.type())

            # Need to continue training on one image until either a stop action is chosen or we have exceeded the
            # maximum number of steps allowed.
            stop_action = False
            step_counter = 0
            previous_state = None

            while((not stop_action) and (step_counter < max_steps)):
                print("The current step is: " + str(step_counter))

                if(step_counter == 0):
                    previous_state = initial_state

                # Get the initial IoU

                # Clear all accumulated gradients
                optimizer.zero_grad()
                # Predict classes using images from the test set
                # torch.unsqueeze(images, 0)
                # print(images.shape)

                # convert the previous state to a tensor
                #print(previous_state)
                previous_state_tensor = torch.Tensor.cuda(torch.from_numpy(np.asarray(previous_state))).float()

                outputs = model.forward(images, previous_state_tensor)

                # Get the predicted action
                _, prediction = torch.max(outputs.data, 1)

                #print(str(labels.numpy().tolist()[0]))

                # Get the label, by taking all actions on previous state
                reward_np = get_np_reward_vector_from_polygon(previous_state, coordinate_action_change_amount,
                labels.numpy().tolist()[0], height_initial, width_initial, coco, step_cost, stop_action_reward)

                reward_tensor = torch.Tensor.cuda(torch.from_numpy(reward_np)).float()

                #print("The shape of the reward tensor is: " + str(reward_tensor.size()))

                # outputs = model.forward(torch.unsqueeze(images, 0))
                height, _, width = labels.shape
                print("The predicted reward is: " + str(outputs))
                print("The actual reward is: " + str(reward_tensor.view(1, 17)))

                loss = loss_fn(outputs, reward_tensor.view(1, 17))
                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                optimizer.step()

                train_loss += loss.cpu().data[0] * images.size(0)
                prediction = int(prediction.data.cpu().numpy()[0])

                if(prediction == 16):
                    stop_action = True

                #print("is prediction 16? " + str(prediction == 16))

                print("The prediction is: " + str(prediction))

                train_acc += torch.sum(outputs.data == reward_tensor.view(1,17).data)

                # Append the reinforcement step counter
                step_counter += 1

                # Make the new state the previous state
                previous_state = apply_action_index_to_state(previous_state, coordinate_action_change_amount,
                                                             prediction, height_initial, width_initial)
                print("The new state is: " + str(previous_state))

        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 2
        train_loss = train_loss / 2

        print("The current epoch is: " + str(epoch))
        #print(train_acc)
        #print(train_loss)

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, train_loss))

    #torch.save(model.state_dict(),
    #           '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/models/'
    #           'reinforcement_learning/rein_' + loss_fn.__str__() + "_" + str(round(train_loss.data.numpy().item(), 5)))

def main():
    train(10)

if __name__ == "__main__":
    main()



