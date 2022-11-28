import numpy as np
import torch.nn as nn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= stride, padding= 1),
            nn.BatchNorm2d(num_features= out_channels),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride= 1, padding= 1),
            nn.BatchNorm2d(out_channels))

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, residual_block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(residual_block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(residual_block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(residual_block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(residual_block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, residual_block, out_planes, num_residual_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes)
            )
        
        layers = []
        layers.append(residual_block(self.inplanes, out_planes, stride, downsample))
        self.inplanes = out_planes
        for i in range(1, num_residual_blocks):
            layers.append(residual_block(self.inplanes, out_planes))

        return nn.Sequential(*layers)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def data_loader(data_dir, batch_size, shuffle, if_train:bool= True, if_download:bool= False):
    # Image transformations
    normalized_image = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.2023, 0.1994, 0.2010],)

    resized_image = transforms.Resize((224,224))
    dataset_save_path = data_dir

    # Configure DataLoaders 
    dataloader = DataLoader(
        dataset=datasets.CIFAR10(
            root=dataset_save_path,
            train= if_train,
            download=if_download,
            transform= transforms.Compose([resized_image,
                                           transforms.ToTensor(),
                                           normalized_image,])
        ),
        batch_size=batch_size, shuffle=shuffle)

    return dataloader

train_dataloader = data_loader(data_dir= "E:\Self Learnings\Data\CIFAR10",
                               batch_size=500,
                               shuffle=False,
                               if_train=True,
                               if_download=True)

test_dataloader = data_loader(data_dir= "E:\Self Learnings\Data\CIFAR10",
                              batch_size= 500,
                              shuffle= False,
                              if_train=False,
                              if_download=False)

def train(model, device, train_dataloader, optim, loss_func, epoch):
    model.train()
    for b_i, (X,y) in enumerate(train_dataloader, start=1):
        X,y = X.to(device), y.to(device)
        optim.zero_grad()
        y_pred = model(X)
        loss = loss_func(y_pred, y)
        loss.backward()
        optim.step()
        if b_i % 10 == 0:
            print("[TRAINING] epoch: {} [{:.2f}/{:.2f}% Completed], training_loss: {:2f}".format(
                  epoch, 
                  b_i*len(X), 
                  len(train_dataloader.dataset), 
                  loss.item()))

def test(model, test_dataloader, device, loss_func, epoch):
    model.eval()
    loss = 0
    with torch.no_grad:
        for b_i, (X,y) in enumerate(test_dataloader, start=1):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_func(y_pred, y).item()
            if b_i % 10 == 0:
                print("[TESTING] epoch: {} [{:.2f}/{:.2f}% Completed], testing_loss: {:2f}".format(
                    epoch, 
                    b_i*len(X), 
                    len(test_dataloader.dataset), 
                    loss))


# Settting Hyper-parameters for ResNet 
num_classes = 10 
num_epochs = 20
batch_size = 16
learning_rate = 0.01

# Defining the model
model = ResNet(residual_block= ResidualBlock, layers=[3,4,6,3])
model = model.to(device=device)

# Loss and optimizer functions 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

# Train the model 
total_steps = len(train_dataloader)

# Training loop
for epoch in range(1, num_epochs):
    train(model=model, device=device, train_dataloader= train_dataloader, optim=optimizer, loss_func=criterion, epoch=epoch)
    test(model=model, test_dataloader=test_dataloader, device=device, loss_func=criterion, epoch=epoch)