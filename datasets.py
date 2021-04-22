from torch.utils.data import DataLoader
import os
import random
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class DogCatDataset(DataLoader):
    def __init__(self):
        dataset_dir = './data/train'
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        train_all_list = os.listdir(dataset_dir)
        self.train_list = []

        dog_sample = random.sample(range(12500), 10000)
        cat_sample = random.sample(range(12500), 10000)

        for i in dog_sample:
            self.train_list.append('{}/dog.{}.jpg'.format(dataset_dir, i))
        for i in cat_sample:
            self.train_list.append('{}/cat.{}.jpg'.format(dataset_dir, i))


    def __getitem__(self, index):
        return self.transform(Image.open(self.train_list[index])), self.train_list[index].split('/')[-1].split('.')[0] == 'dog'


    def __len__(self):
        return len(self.train_list)