import numpy as np
import torch
import torch.nn as nn
from ObjectSegWithRL.src.pytorch_stuff import GregDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

# from ObjectSegWithRL.src.greg_cnn import GregNet
from ObjectSegWithRL.src.models.other.greg_cnn_cSigmoid import GregNet

cuda_avail = torch.cuda.is_available()

model = GregNet(number_of_vertices=15)

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

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.000001)

loss_fn = nn.MSELoss().cuda()
#loss_fn = nn.L1Loss().cuda()

annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/temp2.json'
root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/temp_2/'

#annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/30_vertex_poly_adjusted.json'
#root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/30/'

dataset = GregDataset(annotation_file_path, root_dir_path, transform=test_transformations)

my_dataloader = DataLoader(dataset, batch_size=500, num_workers=4)

def train(num_epochs):
    best_acc = 0.0
    cuda_avail = torch.cuda.is_available()

    train_loss = 0.0

    if cuda_avail:
        model.cuda()
        #print("yes")

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        #train_loss = 0.0
        for i, (images, labels) in enumerate(my_dataloader):
            print(type(my_dataloader))
            print(type(images))
            print(images.numpy().shape)
            #print(str(i))
            # Move images and labels to gpu if available
            mean = np.mean(images.numpy(), axis=(0, 1, 2)).tolist()

            std = np.std(images.numpy(), axis=(0, 1, 2)).tolist()

            t = transforms.Normalize(mean=mean, std=std)

            t(torch.from_numpy(images).view(3,224,224))

            if cuda_avail:
                #print(images.type())
                #print(labels.type())
                #print(images.cuda().type())
                #print(labels.cuda().type())

                #images = torch.Tensor(images)
                #labels = torch.Tensor(labels)
                images = torch.Tensor.cuda((images.cuda())).view(2,3,224,224).float()
                print(images.type())
                labels = torch.Tensor.cuda((labels.cuda())).float()

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            #torch.unsqueeze(images, 0)
            #print(images.shape)

            outputs = model.forward(images)
            #outputs = model.forward(torch.unsqueeze(images, 0))
            # Compute the loss based on the predictions and actual labels
            #print(outputs.type())
            #print(labels.type())
            #print(labels.shape)
            height, _, width = labels.shape
            loss = loss_fn(outputs, labels.view(height, 30))
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data[0] * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            #print(outputs.shape)
            #print(prediction.type())
            #print(labels.data.type())
            #print(labels.shape)
            #print(prediction.shape)
            train_acc += torch.sum(outputs.data == labels.data)
            #train_acc += torch.sum(prediction == labels.data)

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
    train(300)

if __name__ == "__main__":
    main()



