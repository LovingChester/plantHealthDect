import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset, IterableDataset, DataLoader
import time
from tqdm import tqdm
#from config import cfg


class LeafDataset(IterableDataset):

    def __init__(self, data_dir, augmentation, mode, num_workers):
        # data_dir: directory containing leaf data
        # number of workers used to load data
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.mode = mode

        if num_workers <= 0:
           self.num_workers = 1
        else:
           self.num_workers = num_workers

        # transform PIL image to tensor
        self.to_tensor = transforms.ToTensor()

    def process_data(self, data_dir, worker_id):
        count = 0
        image_list = os.listdir(data_dir)
        for image in image_list:
            # get label
            _, label = image.split('-')
            label = int(label.split('.')[0])

            image_path = os.path.join(data_dir, image)
            # converts the image to tensor
            image_data = self.to_tensor(Image.open(image_path).convert('RGB'))

            if count % self.num_workers == worker_id:
                yield image_data, torch.tensor([label])
            
            count += 1
    
    def __len__(self):
        # without: 55448, with: 61486
        # train: 90%, valid: 7%, test: 3%
        if self.augmentation == 'without':
            if self.mode == 'train':
                return int(55448 * 0.9)
            elif self.mode == 'valid':
                return int(55448 * 0.07)
            else:
                return int(55448 * 0.03)
        else:
            if self.mode == 'train':
                return int(61486 * 0.9)
            elif self.mode == 'valid':
                return int(61486 * 0.07)
            else:
                return int(61486 * 0.03)
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
           worker_id = worker.id
           num_workers = worker.num_workers
        else:
           worker_id = 0
           num_workers = 1

        for idx, (image_data, label) in enumerate(self.process_data(self.data_dir, worker_id)):
            yield image_data, label

        #return self.process_data(self.data_dir)

def get_data(cfg):
    train_dataset = LeafDataset(cfg.data.train_dir, cfg.data.augmentation, cfg.data.mode, cfg.train.num_workers)
    valid_dataset = LeafDataset(cfg.data.valid_dir, cfg.data.augmentation, cfg.data.mode, cfg.train.num_workers)
    test_dataset = LeafDataset(cfg.data.test_dir, cfg.data.augmentation, cfg.data.mode, cfg.train.num_workers)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.train.num_workers)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":

    pass

