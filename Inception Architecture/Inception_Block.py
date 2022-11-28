from turtle import forward
import torch
import torch.nn as nn

'''
This class outlines how one Inception Block looks like,
then we will use this block to build complete Inception 
model 
'''

class InceptionModule(nn.Module):
    def __init__(self, input_planes, n_channels1x1, n_channels3x3red, n_channels3x3, n_channels5x5red, n_channels5x5, pooling_planes) -> None:
        super(InceptionModule, self).__init__()
        # 1x1 Conv Branch
        self.block1 = nn.Sequential(
            nn.Conv2d(input_planes, n_channels1x1, kernel_size=1),
            nn.BatchNorm2d(n_channels1x1),
            nn.ReLU(True),
        )

        # 1x1 Convolution -> 3x3 Convolution Branch
        self.block2 = nn.Sequential(
            nn.Conv2d(input_planes, n_channels3x3red, kernel_size=1),
            nn.BatchNorm2d(n_channels3x3red),
            nn.ReLU(True),
            nn.Conv2d(n_channels3x3red, n_channels3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels3x3),
            nn.ReLU(True),
        )

        # 1x1 Convolution -> 3x3 Convolution -> 3x3 Convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(input_planes, n_channels5x5red, kernel_size=1),
            nn.BatchNorm2d(n_channels5x5red),
            nn.ReLU(True),
            nn.Conv2d(n_channels5x5red, n_channels5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels5x5),
            nn.ReLU(True),
            nn.Conv2d(n_channels5x5, n_channels5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels5x5),
            nn.ReLU(True),
        )

        # Maxpool -> 1x1 Convolution
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(input_planes, pooling_planes, kernel_size=1),
            nn.BatchNorm2d(pooling_planes),
            nn.ReLU(True),
        )

    # Forward Pass
    def forward(self, x):
        op1 = self.block1(x)
        op2 = self.block2(x)
        op3 = self.block3(x)
        op4 = self.block4(x)
        return torch.cat([op1,op2,op3,op4], 1)


'''
This class combines multiple Inception Blocks to form one
single model which is called GoogleNet
'''
class GoogLeNet(nn.Module):
    def __init__(self) -> None:
        super(GoogLeNet, self).__init__()
        
        # Initializing the stem of the network 
        self.stem = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.im1 = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.im2 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.im3 = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.im4 = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.im5 = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.im6 = InceptionModule(512, 112, 144, 288, 32,  64,  64)
        self.im7 = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.im8 = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.im9 = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.average_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(4096, 1000)

    
    def forward(self, ip):
        op = self.stem(ip)
        op = self.im1(op)
        op = self.im2(op)
        op = self.max_pool(op)
        '''
        Arrange all the inception blocks in order and return the output'''
        return op
        
        

