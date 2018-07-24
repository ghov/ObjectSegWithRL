# Takes as input a 3 channel tensor and returns a vector
import torch.nn as nn
import torch

class nReLU(nn.ReLU):

    def __init__(self, n, inplace):
        super(nReLU, self).__init__(inplace=inplace)
        #self.n = torch.Tensor(n)
        self.n = n

    def forward(self, x):
        output = super(nReLU, self).forward(x)
        #print(output.shape)
        #print(self.n.expand(output.shape).shape)
        ret_output = self.n * output
        #print(ret_output)
        #ret_output = self.n.expand_as(output) * output
        #print(ret_output.shape)
        return ret_output


        #print(self.n.view(output.shape).shape)
        #ret_output = self.n.view(output.shape) * output
        #return ret_output

class nSigmoid(nn.Sigmoid):

    def __init__(self, n):
        super(nSigmoid, self).__init__()
        self.n = n

    def forward(self, x):
        output = super(nSigmoid, self).forward(x)
        #print(self.n * output)
        return self.n * output


class GregNet(nn.Module):

    def __init__(self, number_of_vertices):
        super(GregNet, self).__init__()

        # This computes the cnn section. After this, we need to resize the features into a vector.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # This take the vector form of the features and computes the linear layers.
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            #nSigmoid(224),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 2 * number_of_vertices),
            #nSigmoid(224),
            #nn.ReLU(inplace=True)
            #nReLU(224, inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.size())
        x = x.view(x.size(0), 256 * 6 * 6)
        #print(x.size())
        x = self.classifier(x)
        return x



