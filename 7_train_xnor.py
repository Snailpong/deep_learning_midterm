import numpy as np
import math
import torch
import random

from tqdm import tqdm
from torch import nn, optim

from models import MyModel7_1, MyModel7_2
from utils import init_device_seed


x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[1], [0], [0], [1], [0], [1], [1], [0]])


def train(model):
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    x_torch = torch.tensor(x).to(device, dtype=torch.float32)
    y_torch = torch.tensor(y).to(device, dtype=torch.float32)

    for i in range(10000):
        y_pred = model(x_torch)
        loss = mse_criterion(y_pred, y_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred_all = model(x_torch)
        print(y_pred_all.cpu().numpy())


if __name__ == '__main__':
    device = init_device_seed()

    train(MyModel7_1().to(device))
    train(MyModel7_2().to(device))