import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
import numpy as np
import os
import argparse
import random
from torchinfo import summary

from resnet18 import ResNet, BasicBlock
# from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='scratch',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision']
)
args = vars(parser.parse_args())

# Set seed.
# seed = 527
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# np.random.seed(seed)
# random.seed(seed)

# learning and training parameters
epochs = 30
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train on the background or the entire iamge
loaders = get_data(background=False, batch_size=batch_size)
train_loader = loaders['train'] 
valid_loader = loaders['valid']

resnet18_pretrained = models.resnet18(weights='IMAGENET1K_V1')

# define model based on the argument parser string
if args['model'] == 'scratch':
    print('[INFO]: Training ResNet18 built from scratch...')
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=39, droprate=0.5).to(device)
    # model.load_state_dict(resnet18_pretrained.state_dict(), strict=False)
    plot_name = 'resnet_scratch'
if args['model'] == 'torchvision':
    print('[INFO]: Training the Torchvision ResNet18 model...')
    model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device) 
    plot_name = 'resnet_torchvision'

# summary of the model used
summary(model=model, 
        input_size=(32, 3, 224, 224),
        col_names=['input_size', "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, 
            valid_loader, 
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
        
    # save the loss and accuracy plots.
    save_plots(
        train_acc, 
        valid_acc, 
        train_loss, 
        valid_loss, 
        name=plot_name
    )

    # save the model state dict
    MODEL_SAVE_PATH = os.path.join('outputs', 'resnet18.pth')
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

    print('TRAINING COMPLETE')

