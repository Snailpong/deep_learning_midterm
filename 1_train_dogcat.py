import os
import torch
import random
import time

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.models as models

from tqdm import tqdm
import numpy as np

from datasets import DogCatDataset
from models import MyModel1


BATCH_SIZE = 32


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    train_loss = []
    val_loss = []
    acc = []

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    dataset = DogCatDataset()
    
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    os.makedirs('./model', exist_ok=True)

    model = MyModel1()
    model.to(device)

    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(10):
        model.train()

        pbar = tqdm(range(len(train_dataloader)))
        pbar.set_description('Epoch {}'.format(epoch+1))

        total_loss = 0.

        for idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            outputs = model(images)
            loss = cls_criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            pbar.set_postfix_str('loss: ' + str(np.around(total_loss / (idx + 1), 4)))
            pbar.update()

        
        with torch.no_grad():
            total_val_loss = 0.
            correct_val = 0.
            for idx, (images, labels) in enumerate(val_dataloader):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                outputs = model(images)
                loss = cls_criterion(outputs, labels)
                total_val_loss += loss.detach().cpu().numpy()

                label_pred = torch.argmax(outputs, 1)
                correct_val += int((labels == label_pred).float().sum())

        train_loss.append(total_loss / len(train_dataloader))
        val_loss.append(total_val_loss / len(val_dataloader))
        acc.append(correct_val / 6000)

        print('\n', train_loss, val_loss, acc)

        torch.save(model.state_dict(), './model/MyModel1')


if __name__ == '__main__':
    train()