import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision.models import vgg16
import time
from tqdm import tqdm
from config import cfg
from data_loaders import get_data

def main():

    # define model
    if cfg.model.name == 'vgg16':
        model = vgg16(weights='DEFAULT')

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    model.to(device)

    # define optimizer
    if cfg.model.name == 'vgg16':
        optim = torch.optim.SGD(model.parameters(),
                            lr=cfg.train.learning_rate,
                            momentum=0.0,
                            weight_decay=cfg.train.l2_reg)

    # get dataloaders
    train_loader, val_loader, _ = get_data(cfg)

    print('starting training')
    model.train()

    for epoch in range(cfg.train.num_epochs):
        # begin training
        loss_train = 0
        for i, (image_data, label) in enumerate(tqdm(val_loader)):
            optim.zero_grad()
            image_data = image_data.to(device)
            label = label.to(device)

            model(image_data)

if __name__ == "__main__":
    main()
