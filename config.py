# This file contains configuration for training and evaluation

from easydict import EasyDict as edict

cfg = edict()

# Model
cfg.model = edict()
# 'vgg16'
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

cfg.train.batch_size = 14
cfg.train.learning_rate = 0.01  # initial learning rate
cfg.train.l2_reg = 0
cfg.train.lr_decay = 0.9
cfg.train.lr_decay_every = 3
cfg.train.shuffle = True
cfg.train.num_epochs = 1
cfg.train.num_workers = 0

cfg.train.out_dir = './outputs/dem_derovative1'
