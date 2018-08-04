from torch.optim import Adam
import torch.nn as nn
import torch
from ObjectSegWithRL.src.reinforce.greg_cnn import GregNet
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from ObjectSegWithRL.src.helper_functions import get_initial_state
from ObjectSegWithRL.src.pytorch_dataset import GregDataset

reward_multiplier = 100
step_cost = -0.5
coordinate_action_change_amount = 10
number_of_actions = 17
polygon_state_length = 8
height_initial = 224
width_initial = 224
max_steps = 100
stop_action_reward = 0.00001
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

loss_fn = nn.MSELoss().cuda()
#loss_fn = nn.L1Loss().cuda()

annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp1.json'
root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_1/'

#annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/30_vertex_poly_adjusted.json'
#root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/30/'

dataset = GregDataset(annotation_file_path, root_dir_path, transform=test_transformations)

my_dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

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

            while((not stop_action) or (step_counter < max_steps)):

                if(step_counter == 0):
                    previous_state = initial_state

                # Get the initial IoU

                # Clear all accumulated gradients
                optimizer.zero_grad()
                # Predict classes using images from the test set
                # torch.unsqueeze(images, 0)
                # print(images.shape)

                outputs = model.forward(images, previous_state)

                # Get the predicted action


                # Get the label, by taking all actions on previous state




                # outputs = model.forward(torch.unsqueeze(images, 0))
                height, _, width = labels.shape
                loss = loss_fn(outputs, labels.view(height, 30))
                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                optimizer.step()

                train_loss += loss.cpu().data[0] * images.size(0)
                _, prediction = torch.max(outputs.data, 1)


                train_acc += torch.sum(outputs.data == labels.data)



                step_counter += 1




        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 5960
        train_loss = train_loss / 5960

        print(epoch)
        print(train_acc)
        print(train_loss)

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, train_loss))

    torch.save(model.state_dict(),
               '/home/greghovhannisyan/PycharmProjects/towards_rlnn_cnn/ObjectSegWithRL/data/models/GregNet_' +
               loss_fn.__str__() + "_" + str(train_loss))

def main():
    train(10)

if __name__ == "__main__":
    main()


