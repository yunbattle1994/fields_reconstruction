'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(CoBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lrl = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.lrl(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.lrl(out)
        return out


class CoBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(CoBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lrl = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.lrl(self.bn1(self.conv1(x)))
        out = self.lrl(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.lrl(out)
        return out


class CoResNet(nn.Module):
    def __init__(self, block, num_blocks, nz=7, nc=1, ndf=16, mode="CY", is_class=False):
        super(CoResNet, self).__init__()
        self.in_planes = 16
        self.nc = nc
        self.ndf = ndf
        # 激活函数在forward中
        self.conv1 = nn.Conv2d(self.nc, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrl = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.ndf , num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.ndf *2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.ndf *4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.ndf *8, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, self.ndf *16, num_blocks[4], stride=2)
        self.layer6 = self._make_layer(block, self.ndf *32, num_blocks[5], stride=2)
        self.layer7 = nn.AvgPool2d(2)
        if mode == "DG":

            self.linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(6144, 1024), nn.ReLU(),
                                        nn.Linear(1024, nz))

        if is_class:
            self.output = nn.Sequential()
        else:
            self.output = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.lrl(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out).squeeze()
        out = self.output(out)
        return out


def CoResNet18(nz=7, nc=1, mode="CY", is_class=False):
    return CoResNet(CoBasicBlock, [1, 1, 2, 2, 1, 1], nc=nc, nz=nz, mode=mode, is_class=is_class)

def CoResNet34(nz=7, nc=1, mode="CY", is_class=False):
    return CoResNet(CoBasicBlock, [2, 3, 3, 3, 3, 2], nc=nc, nz=nz, mode=mode, is_class=is_class)

def CoResNet50(nz=7, nc=1, ndf=16, is_class=False):
    return CoResNet(CoBasicBlock, [2, 4, 6, 6, 4, 2], nz=nz, is_class=is_class)

def CoResNet101(nz=7, nc=1, ndf=16, is_class=False):
    return CoResNet(CoBasicBlock, [3, 10, 12, 12, 10, 3], nz=nz, is_class=is_class)

def CoResNet152(nz=7, nc=1, ndf=16, is_class=False):
    return CoResNet(CoBasicBlock, [3, 12, 23, 23, 12, 3], nz=nz, is_class=is_class)


if __name__ == '__main__':

    net = CoResNet34(nc=4, nz=4, mode="DG", is_class=True)
    # y = net(torch.randn(10, 1, 792, 40))
    # y = net(torch.randn(10, 1, 792, 40))
    y = net(torch.randn(10, 4, 792, 40))
    print(y.size())

# test()
