import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset, IterableDataset, DataLoader
import time
from tqdm import tqdm
from config import cfg


class LeafDataset(IterableDataset):

    def __init__(self, data_dir, augmentation, model, mode, num_workers):
        # data_dir: directory containing leaf data
        # number of workers used to load data
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.model = model
        self.mode = mode

        if num_workers <= 0:
           self.num_workers = 1
        else:
           self.num_workers = num_workers

        # transform to tensor, resize to (256, 256) and center crop to (224, 224)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((256, 256), antialias=True),
                                             transforms.CenterCrop(224)])
        
        if self.augmentation == 'without':
            self.normalize = transforms.Normalize(mean=[0.4685, 0.5424, 0.4491],
                                                std=[0.2337, 0.2420, 0.2531])
        elif self.augmentation == 'with':
            self.normalize = transforms.Normalize(mean=[0.4683, 0.5414, 0.4477],
                                                std= [0.2327, 0.2407, 0.2521])

    def process_data(self, data_dir, worker_id):
        count = 0
        image_list = os.listdir(data_dir)
        for image in image_list:
            # get label
            _, label = image.split('-')
            label = int(label.split('.')[0])
            image_path = os.path.join(data_dir, image)

            # converts the image to tensor, resize image to (256, 256) and center it to (224, 224)
            image_data = self.transform(Image.open(image_path).convert('RGB'))

            # normalize image
            #image_data = self.normalize(image_data)

            if count % self.num_workers == worker_id:
                yield image_data, torch.tensor(label)
            
            count += 1
    
    def __len__(self):
        # without: 55448, with: 61486
        # train: 90%, valid: 7%, test: 3%
        if self.augmentation == 'without':
            if self.mode == 'train':
                return int(55448 * 0.9)
            elif self.mode == 'valid':
                return int(55448 * 0.07)
            elif self.mode == 'test':
                return int(55448 * 0.03)
        else:
            if self.mode == 'train':
                return int(61486 * 0.9)
            elif self.mode == 'valid':
                return int(61486 * 0.07)
            elif self.mode == 'test':
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

    if cfg.data.augmentation == 'without':
        normalize = transforms.Normalize(mean=[0.4685, 0.5424, 0.4491],
                                        std=[0.2337, 0.2420, 0.2531])
    elif cfg.data.augmentation == 'with':
        normalize = transforms.Normalize(mean=[0.4683, 0.5414, 0.4477],
                                        std= [0.2327, 0.2407, 0.2521])

    transform = transforms.Compose([transforms.Resize((256, 256), antialias=True),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(), normalize])
    
    train_dataset = datasets.ImageFolder(root=cfg.data.train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(root=cfg.data.valid_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=cfg.data.test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":

    train_loader, valid_loader, test_loader = get_data(cfg)
    for i, (image_data, label) in enumerate(tqdm(train_loader)):
        if i == 5: break
        to_pil = transforms.ToPILImage()
        plt.imshow(to_pil(image_data[0]))
        plt.show()
