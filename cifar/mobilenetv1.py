import torch.nn as nn

import torch.nn.functional as F

def Conv2dBNReLU(in_channels,out_channels , stride,kernel_size,padding,groups=1):
    se = nn.Sequential()
    conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,groups=groups,padding=padding)
    bn = nn.BatchNorm2d(num_features=out_channels)
    se.add_module('conv',conv)
    se.add_module('bn',bn)
    se.add_module('relu',nn.ReLU())
    return se

class MobileV1Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self,  in_planes, out_planes, stride=1):
        super(MobileV1Block, self).__init__()
        self.depthwise = Conv2dBNReLU(in_channels=in_planes, out_channels=2*in_planes, kernel_size=3,
                                          stride=stride, padding=1, groups=in_planes)
        self.pointwise = Conv2dBNReLU(in_channels=2*in_planes, out_channels=out_planes, kernel_size=1,
                                          stride=1, padding=0)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
