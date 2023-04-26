import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayerCifar(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayerCifar, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockCifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockCifar, self).__init__()

        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                tup1 = (0, 0, 0, 0, planes // 4, planes // 4)
                def f(x): return F.pad(x[:, :, ::2, ::2], tup1, 'constant', 0)
                self.shortcut = LambdaLayerCifar(f)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3,
                               16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet14(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.left_model = ResNetCifar(BasicBlockCifar, [2, 2, 2])
        self.right_model = ResNetCifar(BasicBlockCifar, [2, 2, 2])

        self.left_model.conv1 = nn.Conv2d(inc,
                                          16,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=False)

        self.right_model.conv1 = nn.Conv2d(inc,
                                           16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False)

        self.left_model.fc = nn.Linear(64, 32, bias=True)
        self.left_dropout = nn.Dropout(p=0.5)

        self.right_model.fc = nn.Linear(64, 32, bias=True)
        self.right_dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(64, outc, bias=True)

    def forward(self, left_eye, right_eye):
        left_out = self.left_dropout(self.left_model(left_eye))
        right_out = self.right_dropout(self.right_model(right_eye))

        out = self.fc(torch.cat((left_out, right_out), dim=1))
        return out


class ResNet20(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()

        self.left_model = ResNetCifar(BasicBlockCifar, [3, 3, 3])
        self.right_model = ResNetCifar(BasicBlockCifar, [3, 3, 3])

        self.left_model.conv1 = nn.Conv2d(inc,
                                          16,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=False)

        self.right_model.conv1 = nn.Conv2d(inc,
                                           16,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False)

        self.left_model.fc = nn.Linear(64, 32, bias=True)
        self.left_dropout = nn.Dropout()

        self.right_model.fc = nn.Linear(64, 32, bias=True)
        self.right_dropout = nn.Dropout()

        self.fc = nn.Linear(64, outc, bias=True)
        self.relu = nn.ReLU()

    def forward(self, left_eye, right_eye):
        left_out = self.left_dropout(self.left_model(left_eye))
        right_out = self.right_dropout(self.right_model(right_eye))

        out = self.relu(self.fc(torch.cat((left_out, right_out), dim=1)))
        return out
