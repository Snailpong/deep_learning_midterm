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