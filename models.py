import torch
from torch import nn

import torchvision.models as models


class MyModel1(torch.nn.Module):
    def __init__(self):
        super(MyModel1, self).__init__()
        self.vgg16 = models.vgg16_bn(pretrained=True)
        self.output = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.vgg16(x)
        return self.output(x)


class MyModel5(torch.nn.Module):
    def __init__(self):
        super(MyModel5, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        return self.layers(x)


class MyModel7_1(torch.nn.Module):
    def __init__(self):
        super(MyModel7_1, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


class MyModel7_2(torch.nn.Module):
    def __init__(self):
        super(MyModel7_2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.layers(x)