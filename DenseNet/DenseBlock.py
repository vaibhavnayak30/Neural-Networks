import torch
import torch.nn as nn
import torch.functional as F

'''
Class Dense Block and define forwards which takes inputs and 
after introducing non-linearity in the input concats all the layers
'''
class DenseBlock(nn.Module):
    def __init__(self, input_num_planes, rate_inc):
        super(DenseBlock, self).__init__()
        self.conv_layer1 = nn.Conv2d(input_num_planes, 4*rate_inc, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(input_num_planes)
        self.conv_layer2 = nn.Conv2d(4*rate_inc, rate_inc, kernel_size=3, padding=1, bias=False)

    def forward(self, inp):
        op = self.conv_layer1(F.relu(self.batch_norm1(inp)))
        op = self.conv_layer2(F.relu(self.batch_norm2(op)))
        op = torch.cat((inp,op), 1)
        return op

'''
Class Transition Block which keeps a check on number of layers entering
next Dense Layer by using 1x1 Conv operation
'''
class TransBlock(nn.Module):
    def __init__(self, input_num_planes, output_num_planes):
        super(TransBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(input_num_planes)
        self.conv_layer = nn.Conv2d(input_num_planes, output_num_planes, kernel_size=1, bias=False)

    def forward(self, inp):
        op = self.conv_layer(F.relu(self.batch_norm(inp)))
        op = F.avg_pool2d(op,2)
        return op
        
