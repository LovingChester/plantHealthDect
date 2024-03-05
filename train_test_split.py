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
images = os.listdir(train_dir)
for image in images:
    path = os.path.join(train_dir, image)
    os.remove(path)

images = os.listdir(valid_dir)
for image in images:
    path = os.path.join(valid_dir, image)
    os.remove(path)

images = os.listdir(test_dir)
for image in images:
    path = os.path.join(test_dir, image)
    os.remove(path)


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

    for i, image in enumerate(tqdm(image_list)):
        image_path = os.path.join(file_path, image)
        if i in train:
            image_dest_path = os.path.join(train_dir, "train"+str(train_size)+"-"+str(label)+".jpg")
            copyfile(image_path, image_dest_path)
            train_size += 1
        elif i in valid:
            image_dest_path = os.path.join(valid_dir, "valid"+str(valid_size)+"-"+str(label)+".jpg")
            copyfile(image_path, image_dest_path)
            valid_size += 1
        else:
            image_dest_path = os.path.join(test_dir, "test"+str(test_size)+"-"+str(label)+".jpg")
            copyfile(image_path, image_dest_path)
            test_size += 1
    