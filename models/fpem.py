'''
This is is FPEM module for PAN.
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

__all__ = ['Conv_BN_ReLU','FPEM']

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FPEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM, self).__init__()
        planes = out_channels
        self.dwconv3_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        f3 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3, f2)))
        f1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2, f1)))

        f2 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2, f1)))
        f3 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3, f2)))
        f4 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3)))

        return f1, f2, f3, f4

# unit testing
if __name__ == '__main__':

    batch_size = 32
    Height = 512
    Width = 512
    Channel = 128

    f1 = torch.randn(batch_size,Channel,Height//4,Width//4)
    f2 = torch.randn(batch_size,Channel,Height//8,Width//8)
    f3 = torch.randn(batch_size,Channel,Height//16,Width//16)
    f4 = torch.randn(batch_size,Channel,Height//32,Width//32)
    print("Input of FPEM layer 1:", f1.shape)
    print("Input of FPEM layer 2:", f2.shape)
    print("Input of FPEM layer 3:", f3.shape)
    print("Input of FPEM layer 4:", f4.shape)

    fpem_model = FPEM(Channel, Channel)

    f1, f2, f3, f4 = fpem_model(f1, f2, f3, f4)
    print("Output of FPEM layer 1:", f1.shape)
    print("Output of FPEM layer 2:", f2.shape)
    print("Output of FPEM layer 3:", f3.shape)
    print("Output of FPEM layer 4:", f4.shape)