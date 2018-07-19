from torch.optim import Adam
import torch.nn as nn
import torch
from ObjectSegWithRL.src.greg_cnn import GregNet
from ObjectSegWithRL.src.pytorch_stuff import GregDataset
from torchvision import transforms

cuda_avail = torch.cuda.is_available()

model = GregNet(number_of_vertices=15)

#if cuda_avail:
#    model.cuda()

test_transformations = transforms.Compose([
    transforms.ToTensor()

])
# test_transformations = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#
# ])

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

loss_fn = nn.L1Loss()

annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/30_vertex_poly_adjusted.json'
root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/30/'

dataset = GregDataset(annotation_file_path, root_dir_path, transform=test_transformations)

def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(dataset):
            print(str(i))
            # Move images and labels to gpu if available
            if cuda_avail:
                images = torch.Tensor(images)
                labels = torch.Tensor(labels)
                #images = torch.Tensor(images.cuda())
                #labels = torch.Tensor(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            #torch.unsqueeze(images, 0)
            outputs = model.forward(torch.unsqueeze(images, 0))
            #outputs = model.forward(images.unequeze_(0))
            # Compute the loss based on the predictions and actual labels
            print(outputs.type())
            print(labels.type())
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data[0] * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            #print(prediction.type())
            #print(labels.data.type())
            #train_acc += torch.sum(prediction == labels.data)

        # Compute the average acc and loss over all 50000 training images
        #train_acc = train_acc / 5960
        train_loss = train_loss / 5960

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, train_loss))

def main():
    train(1)

if __name__ == "__main__":
    main()



