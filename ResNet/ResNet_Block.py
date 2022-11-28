import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, input_num_planes, num_planes, stride= 1):
        super(BasicBlock, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels= input_num_planes, out_channels= num_planes, kernel_size= 3, 
                                     stride= stride, padding= 1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=num_planes)
        self.conv_layer2 = nn.Conv2d(in_channels=num_planes, out_channels= num_planes, kernel_size=3, stride=1,
                                     padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=num_planes)
        self.res_connection = nn.Sequential(
            nn.Conv2d(in_channels= input_num_planes,)
        )
 