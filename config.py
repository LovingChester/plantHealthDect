# This file contains configuration for training and evaluation

from easydict import EasyDict as edict

cfg = edict()

# Model
cfg.model = edict()
# 'vgg16', 'vit'
cfg.model.name = 'vgg16'
cfg.model.pre_trained = False

# Data
# without: 55448, with: 61486
cfg.data = edict()
cfg.data.augmentation = 'without' # with or without
cfg.data.data_dir = '../Plant_leave_diseases_dataset_' + cfg.data.augmentation + '_augmentation'
cfg.data.train_dir = '../Plant_leave_diseases_split_' + cfg.data.augmentation + '_augmentation/train'
cfg.data.valid_dir = '../Plant_leave_diseases_split_' + cfg.data.augmentation + '_augmentation/valid'
cfg.data.test_dir = '../Plant_leave_diseases_split_' + cfg.data.augmentation + '_augmentation/test'

cfg.data.mode = 'train'

# Training details
cfg.train = edict()

cfg.train.batch_size = 16
cfg.train.learning_rate = 0.01  # initial learning rate
cfg.train.l2_reg = 0
cfg.train.lr_decay = 0.9
cfg.train.lr_decay_every = 3
cfg.train.shuffle = True
cfg.train.num_epochs = 2
cfg.train.num_workers = 2

cfg.train.weights_without_aug = [44.01, 44.64, 100.81, 16.85, 24.26, 18.46, 32.46, 26.35, 54.04, 23.26, 23.86, 28.15, 
                                23.49, 20.05, 65.54, 25.77, 5.03, 12.07, 77.01, 27.81, 18.76, 27.72, 182.39, 27.72, 
                                74.73, 5.45, 15.11, 60.8, 25.0, 13.03, 27.72, 17.43, 14.52, 29.12, 15.65, 16.54, 19.75, 
                                74.33, 5.18]
cfg.train.weights_with_aug = [30.74, 30.74, 30.74, 18.69, 26.9, 20.47, 30.74, 29.22, 30.74, 25.79, 26.46, 30.74, 26.05, 
                              22.23, 30.74, 28.57, 5.58, 13.38, 30.74, 30.74, 20.8, 30.74, 30.74, 30.74, 30.74, 6.04, 
                              16.75, 30.74, 27.72, 14.45, 30.74, 19.32, 16.1, 30.74, 17.36, 18.34, 21.9, 30.74, 5.74]

cfg.train.out_dir = './outputs'
