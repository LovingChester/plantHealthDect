import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from config import cfg

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
        print(label)
        if i == 5: break
        to_pil = transforms.ToPILImage()
        plt.imshow(to_pil(image_data[0]))
        plt.show()
