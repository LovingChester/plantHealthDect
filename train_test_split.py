# This file will create training, validation and test data

from config import cfg
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

train_dir = cfg.data.train_dir
valid_dir = cfg.data.valid_dir
test_dir = cfg.data.test_dir

# create train, valid and test directory
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# delete image before putting in any
image_labels = os.listdir(train_dir)
for label in image_labels:
    path = os.path.join(train_dir, label)
    images = os.listdir(path)
    for image in images:
        image_path = os.path.join(path, image)
        os.remove(image_path)

image_labels = os.listdir(valid_dir)
for label in image_labels:
    path = os.path.join(valid_dir, label)
    images = os.listdir(path)
    for image in images:
        image_path = os.path.join(path, image)
        os.remove(image_path)

image_labels = os.listdir(test_dir)
for label in image_labels:
    path = os.path.join(test_dir, label)
    images = os.listdir(path)
    for image in images:
        image_path = os.path.join(path, image)
        os.remove(image_path)


# train: 90%, valid: 7%, test: 3%
data_dir = cfg.data.data_dir
files = os.listdir(data_dir)
train_size, valid_size, test_size = 0, 0, 0
for label, file in enumerate(files):
    file_path = os.path.join(data_dir, file)
    image_list = os.listdir(file_path)

    indices = list(range(0, len(image_list)))
    # split into train and valid_test first
    train, valid_test = train_test_split(indices, test_size=0.1)
    # split valid_test into valid and test
    valid, test = train_test_split(valid_test, test_size=0.3)
    # convert to set
    train, valid, test = set(train), set(valid), set(test)

    # create subfile for each class
    train_sub = os.path.join(train_dir, file)
    if not os.path.exists(train_sub):
        os.makedirs(train_sub)

    valid_sub = os.path.join(valid_dir, file)
    if not os.path.exists(valid_sub):
        os.makedirs(valid_sub)

    test_sub = os.path.join(test_dir, file)
    if not os.path.exists(test_sub):
        os.makedirs(test_sub)

    for i, image in enumerate(tqdm(image_list)):
        image_path = os.path.join(file_path, image)
        if i in train:
            image_dest_path = os.path.join(train_sub, "train"+str(train_size)+".jpg")
            train_size += 1
        elif i in valid:
            image_dest_path = os.path.join(valid_sub, "valid"+str(valid_size)+".jpg")
            valid_size += 1
        else:
            image_dest_path = os.path.join(test_sub, "test"+str(test_size)+".jpg")
            test_size += 1
        
        copyfile(image_path, image_dest_path)
    