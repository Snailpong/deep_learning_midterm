import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import random

from tqdm import tqdm
from torch import nn, optim

from models import MyModel5
from utils import init_device_seed


def get_fx(x):
    return 0.2 + 0.4 * (x ** 2) + 0.3 * x * np.sin(15 * x) + 0.05 * np.cos(40 * x)


def plot_graph(x, y):
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    device = init_device_seed()

    x = np.linspace(0, 1, 101)
    y = get_fx(x)
    # plot_graph(x, y)

    pred_list = np.empty((100, 101))

    for t in tqdm(range(100)):
        model = MyModel5().to(device)
        mse_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model.train()

        for i in range(10000):
            x_sample = np.random.choice(x, 64)
            y_sample = get_fx(x_sample)

            x_sample = torch.from_numpy(x_sample[:, np.newaxis]).to(device, dtype=torch.float32)
            y_sample = torch.from_numpy(y_sample[:, np.newaxis]).to(device, dtype=torch.float32)

            y_pred = model(x_sample)
            loss = mse_criterion(y_pred, y_sample)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred_all = model(torch.from_numpy(x[:, np.newaxis]).to(device, dtype=torch.float32))
            pred_list[t] = y_pred_all.cpu().numpy().ravel()
            # plot_graph(x, y_pred_all.cpu().numpy().ravel())
    
    pred_mean = np.mean(pred_list, axis=0)
    pred_std = np.std(pred_list, axis=0)
    print(pred_mean, pred_std)

    plt.plot(x, y, marker='.', linestyle='None')
    plt.errorbar(x, pred_mean, yerr=pred_std)
    plt.show()
