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

annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp2.json'
root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_2/'

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
        #train_loss = 0.0
        for i, (images, labels) in enumerate(my_dataloader):

            print(type(my_dataloader))
            print(type(images))
            print(images.numpy().shape)
            # print(str(i))
            # Move images and labels to gpu if available

            # Calculate the mean
            mean = np.mean(images.numpy(), axis=(0, 1, 2)).tolist()

            # Calculate the standard deviation
            std = np.std(images.numpy(), axis=(0, 1, 2)).tolist()

            # Construct the normalizatin object with the above mean and standard deviation
            t = transforms.Normalize(mean=mean, std=std)

            # Don't know what this is
            t(torch.from_numpy(images).view(3, 224, 224))

            if cuda_avail:
                #print(images.type())
                #print(labels.type())
                #print(images.cuda().type())
                #print(labels.cuda().type())

                #images = torch.Tensor(images)
                #labels = torch.Tensor(labels)
                images = torch.Tensor.cuda((images.cuda())).view(2, 3, 224, 224).float()
                print(images.type())
                #labels = torch.Tensor.cuda((labels.cuda())).float()



            # If



            # Need to continue training on one image until either a stop action is chosen or we have exceeded the
            # maximum number of steps allowed.
            stop_action = False
            step_counter = 0

            while((not stop_action) or (step_counter < max_steps)):

                # Clear all accumulated gradients
                optimizer.zero_grad()
                # Predict classes using images from the test set
                # torch.unsqueeze(images, 0)
                # print(images.shape)

                outputs = model.forward(images, previous_state)
                # outputs = model.forward(torch.unsqueeze(images, 0))
                # Compute the loss based on the predictions and actual labels
                # print(outputs.type())
                # print(labels.type())
                # print(labels.shape)
                height, _, width = labels.shape
                loss = loss_fn(outputs, labels.view(height, 30))
                # Backpropagate the loss
                loss.backward()

                # Adjust parameters according to the computed gradients
                optimizer.step()

                train_loss += loss.cpu().data[0] * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                # print(outputs.shape)
                # print(prediction.type())
                # print(labels.data.type())
                # print(labels.shape)
                # print(prediction.shape)
                train_acc += torch.sum(outputs.data == labels.data)
                # train_acc += torch.sum(prediction == labels.data)



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



