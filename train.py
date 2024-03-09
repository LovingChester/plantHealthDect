import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import vgg16, vit_b_16
import time
import sys
from tqdm import tqdm
from config import cfg
from data_loaders import get_data

def main():

    # define model
    if cfg.model.name == 'vgg16':
        model = vgg16(weights='DEFAULT')
        model.classifier[6] = nn.Linear(4096, 39, bias=True)
    elif cfg.model.name == 'vit':
        model = vit_b_16(weights='DEFAULT')
        model.heads.head = nn.Linear(768, 39, bias=True)

    print(model)

    for param in model.parameters():
        param.requires_grad = True

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    model.to(device)

    # define optimizer
    if cfg.model.name == 'vgg16':
        optim = torch.optim.SGD(model.parameters(),
                            lr=cfg.train.learning_rate,
                            momentum=0.0,
                            weight_decay=cfg.train.l2_reg)
    elif cfg.model.name == 'vit':
        optim = torch.optim.SGD(model.parameters(),
                            lr=cfg.train.learning_rate,
                            momentum=0.9,
                            weight_decay=cfg.train.l2_reg)

    # get dataloaders
    train_loader, val_loader, _ = get_data(cfg)

    # loss function
    if cfg.data.augmentation == 'without':
        weight = torch.tensor(cfg.train.weights_without_aug).to(device)
    else:
        weight = torch.tensor(cfg.train.weights_with_aug).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_val_loss = sys.maxsize

    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(
            'output directory ', out_dir,
            ' already exists. Make sure you are not overwriting previously trained model...'
        )

    train_loss_log = []
    val_loss_log = []

    print('starting training')

    for epoch in range(cfg.train.num_epochs):
        # begin training
        model.train()
        cfg.data.mode = 'train'
        loss_train = 0
        for i, (image_data, label) in enumerate(tqdm(train_loader)):
            optim.zero_grad()
            image_data = image_data.to(device)
            label = label.to(device)

            output = model(image_data)

            loss = criterion(output, label)
            loss.backward()
            loss_train += loss.item()

            optim.step()
        
        # end of training for this epoch
        loss_train /= len(train_loader)
        train_loss_log.append(loss_train)

        # begin validation
        loss_val = 0
        model.eval()
        cfg.data.mode = 'valid'
        with torch.no_grad():
            for i, (image_data, label) in enumerate(val_loader):
                optim.zero_grad()  # clear gradients

                image_data = image_data.to(device)
                label = label.to(device)

                output = model(image_data)

                loss = criterion(output, label)
                loss_val += loss.item()
        
        # end of validation
        loss_val /= len(val_loader)
        val_loss_log.append(loss_val)

        print('End of epoch ', epoch + 1, ' , Train loss: ', loss_train,
                ', val loss: ', loss_val)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            fname = 'model_dict.pth'
            torch.save(model.state_dict(), os.path.join(out_dir, fname))
            print('================= model saved at epoch: ', epoch + 1,
                  ' =================')
    

    fname = 'model_dict_final.pth'
    torch.save(model.state_dict(), os.path.join(out_dir, fname))
    print('================= model saved at the end of the training =================')

    # save loss curves
    plt.figure()
    plt.plot(train_loss_log)
    plt.plot(val_loss_log)
    plt.legend(['train loss', 'test loss'])
    fname = os.path.join(out_dir, 'loss.png')
    plt.savefig(fname)

    param = os.path.join(out_dir, 'parameters.txt')
    with open(param, 'w') as result_file:
        result_file.write(str(cfg))

if __name__ == "__main__":
    main()
