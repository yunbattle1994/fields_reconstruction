'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class UNFlatten(nn.Module):

    def __init__(self, planes, height=25, weight=2):
        super(UNFlatten, self).__init__()
        self.height = height
        self.weight = weight
        self.planes = planes

    def forward(self, x):
        return x.reshape((x.size(0), self.planes, self.height, self.weight))

class DeBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_pad=0):
        super(DeBasicBlock, self).__init__()
        if out_pad == 0:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                            output_padding=0, bias=False)
            self.conv_shortcut = nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1,
                                            output_padding=0, bias=False)
        elif out_pad == 0.5:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                            output_padding=(1,0), bias=False)
            self.conv_shortcut = nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1,
                                            output_padding=(1,0), bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
            self.conv_shortcut = nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                self.conv_shortcut,
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


class DeResNet(nn.Module):
    def __init__(self, block, num_blocks, nz=7, nc=1, ngf=16):
        super(DeResNet, self).__init__()
        self.in_planes = 16 * 32
        self.nc = nc
        self.ngf = ngf
        self.linear = nn.Linear(nz, self.ngf * 32 * block.expansion * 25 * 2)

        self.layer1 = self._make_layer(block, self.ngf * 32, num_blocks[0], stride=2, out_pad=0.5)
        self.layer2 = self._make_layer(block, self.ngf * 16, num_blocks[1], stride=2, out_pad=0)
        self.layer3 = self._make_layer(block, self.ngf * 8, num_blocks[2], stride=2, out_pad=1)
        self.layer4 = self._make_layer(block, self.ngf * 4, num_blocks[3], stride=2, out_pad=1)
        self.layer5 = self._make_layer(block, self.ngf * 2, num_blocks[4], stride=2, out_pad=1)
        self.layer6 = self._make_layer(block, self.ngf, num_blocks[5], stride=1, out_pad=0)


        self.conv1 = nn.ConvTranspose2d(self.in_planes, self.nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.nc)
        self.uf = UNFlatten(planes=32*self.ngf, weight=25, height=2)


    def _make_layer(self, block, planes, num_blocks, stride, out_pad):
        strides = [stride] + [1]*(num_blocks-1)
        out_pads = [out_pad] + [0]*(num_blocks-1)
        layers = []
        for stride,out_pad in zip(strides, out_pads):
            layers.append(block(self.in_planes, planes, stride, out_pad))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(out.size(0), 32*self.ngf, 25, 2)
        # out = self.uf(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.conv1(out)

        return out

    def extract(self, x):
        out0 = self.linear(x)
        out1 = out0.reshape(out0.size(0), 32 * self.ngf, 25, 2)
        # out = self.uf(out)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out6 = self.layer5(out5)
        out7 = self.layer6(out6)
        out = self.conv1(out7)

        extract = [out0, out1, out2, out3, out4, out5, out6, out7, out]

        return extract


def DeResNet18(nz=7, nc=1, ngf=16):
    return DeResNet(DeBasicBlock, [1, 1, 2, 2, 1, 1], nz=nz, nc=nc)

def DeResNet34(nz=7, nc=1, ngf=16):
    return DeResNet(DeBasicBlock, [2, 3, 3, 3, 3, 2], nz=nz, nc=nc)

def DeResNet50(nz=7, nc=1, ngf=16):
    return DeResNet(DeBasicBlock, [2, 4, 6, 6, 4, 2], nz=nz)

def DeResNet101(nz=7, nc=1, ngf=16):
    return DeResNet(DeBasicBlock, [3, 10, 12, 12, 10, 3], nz=nz)

def DeResNet152(nz=7, nc=1, ngf=16):
    return DeResNet(DeBasicBlock, [3, 12, 23, 23, 12, 3], nz=nz)


if __name__ == '__main__':

    net = DeResNet34(nz=20, nc=3, ngf=16)
    print(net)
    y = net(torch.randn(1, 20))
    print(y.size())

# test()
