# import required libraries
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



torch.manual_seed(0)

class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet,self).__init__()
        # 3 input image channel, 6 output feature maps, 5x5 conv kernel
        self.cn1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        # 6 input image channels, 16 output feature maps, 5x5 conv kernel
        self.cn2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # fully connected layers of size 120, 84 and 10
        self.fc1 = nn.Linear(in_features= 16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Convolution with a 5x5 kernel
        x = F.relu(self.cn1(x))
        # max pooling over a 2x2 window
        x = F.max_pool2d(x, (2,2))
        # Convolution with a 5x5 kernel 
        x = F.relu(self.cn2(x))
        # Max pooling over a 2x2 window
        x = F.max_pool2d(x, (2,2))
        # Flatter the image 
        x = x.view(-1, self.flattened_features(x=x))
        # Fully connected operations 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def flattened_features(self,x):
        # all except the first (batch) dimension
        size = x.size()[1:]
        num_feats = 1
        for s in size:
            num_feats *= s
        return num_feats


def train(net, device, trainLoader, optim, epoch):
    # initialize the loss 
    loss_total = 0

    for i, data in enumerate(trainLoader):
        # ip refers to input image and ground_truth refers to the output class that image belong to 
        ip, ground_truth = data

        # Load input parameters to device
        ip, ground_truth = ip.to(device), ground_truth.to(device)

        # zero out parameters gradients 
        optim.zero_grad()

        # forward pass + loss calculation + backward pass 
        op = net(ip)
        loss =  F.cross_entropy(op, ground_truth)
        loss.backward()
        optim.step()

        # update loss 
        loss_total += loss.item()

        # print loss statistics for every 10 
        if i % 10 == 0:
            print("Epoch:{} [{}/{} ({:.0f}%)]\t Training Loss: {:.2f} \t Total Loss: {:.2f}".format(
                epoch, i * len(ip), len(trainLoader.dataset),
                100 * (i * len(ip) / len(trainLoader.dataset)), loss.item(), loss_total
            ))

def test(net, device, testLoader):
    counter = 0
    success = 0
    with torch.no_grad():
        for data in testLoader:
            im, ground_truth = data
            
            # Load data to device
            im, ground_truth = im.to(device), ground_truth.to(device)

            op = net(im)
            _, pred = torch.max(op.data, 1)
            counter += ground_truth.size(0)
            success += (pred == ground_truth).sum().item()

    print("LeNet accuracy on 10000 images from test dataset: {} %".format(100 * success / counter))


# Train Data Set and Loader
train_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32,4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,0.5,0.5),
                                                            std= (0.5,0.5,0.5))
                                    ])

trainset = torchvision.datasets.CIFAR10(root='/Users/vaibhav/Desktop/Exercise Data/CIFAR10', 
                                        train = True,
                                        download= True,
                                        transform = train_transform)

trainLoader = DataLoader(dataset= trainset, batch_size= 8, shuffle= True, num_workers= 1)


# Test Data Set and Loader 
test_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

testset = torchvision.datasets.CIFAR10(root= '/Users/vaibhav/Desktop/Exercise Data/CIFAR10',
                                       train= False,
                                       download= True,
                                       transform= test_transform)

testLoader = DataLoader(dataset= testset, batch_size= 10000, shuffle=True, num_workers= 2)

# ordering is important
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define optimiser 
optim = torch.optim.Adam(lenet.parameters(), lr=0.001)

#training loop over dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the model
lenet = LeNet()
lenet = lenet.to(device)
for epoch in range(50):
    train(net=lenet, device=device, trainLoader=trainLoader, optim=optim, epoch=epoch)
    test(net=lenet, device=device, testLoader=testLoader)

print("Finished Training")