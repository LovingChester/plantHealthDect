# This file will split leaf and non-leaf images

from config import cfg
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm

train_dir = cfg.data.leaf_train_dir
valid_dir = cfg.data.leaf_valid_dir
test_dir = cfg.data.leaf_test_dir

# create train, valid and test directory
train_dir_non_leaf = os.path.join(train_dir, 'non_leaf')
train_dir_leaf = os.path.join(train_dir, 'leaf')
if not os.path.exists(train_dir_non_leaf):
    os.makedirs(train_dir_non_leaf)
    os.makedirs(train_dir_leaf)

valid_dir_non_leaf = os.path.join(valid_dir, 'non_leaf')
valid_dir_leaf = os.path.join(valid_dir, 'leaf')
if not os.path.exists(valid_dir_non_leaf):
    os.makedirs(valid_dir_non_leaf)
    os.makedirs(valid_dir_leaf)

test_dir_non_leaf = os.path.join(test_dir, 'non_leaf')
test_dir_leaf = os.path.join(test_dir, 'leaf')
if not os.path.exists(test_dir_non_leaf):
    os.makedirs(test_dir_non_leaf)
    os.makedirs(test_dir_leaf)

# delete image before putting in any
images = os.listdir(train_dir_non_leaf)
for image in images:
    image_path = os.path.join(train_dir_non_leaf, image)
    os.remove(image_path)

images = os.listdir(train_dir_leaf)
for image in images:
    image_path = os.path.join(train_dir_leaf, image)
    os.remove(image_path)

images = os.listdir(valid_dir_non_leaf)
for image in images:
    image_path = os.path.join(valid_dir_non_leaf, image)
    os.remove(image_path)

images = os.listdir(valid_dir_leaf)
for image in images:
    image_path = os.path.join(valid_dir_leaf, image)
    os.remove(image_path)

images = os.listdir(test_dir_non_leaf)
for image in images:
    image_path = os.path.join(test_dir_non_leaf, image)
    os.remove(image_path)

images = os.listdir(test_dir_leaf)
for image in images:
    image_path = os.path.join(test_dir_leaf, image)
    os.remove(image_path)

# train: 90%, valid: 7%, test: 3%
data_dir = cfg.data.data_dir

# split the non leaf data first
non_leaf_dir = os.path.join(data_dir, 'Background_without_leaves')
non_leaf_list = os.listdir((non_leaf_dir))
indices = list(range(0, len(non_leaf_list)))
# split into train and valid_test first
train, valid_test = train_test_split(indices, test_size=0.1)
# split valid_test into valid and test
valid, test = train_test_split(valid_test, test_size=0.3)
# convert to set
train, valid, test = set(train), set(valid), set(test)

for i, image in enumerate(non_leaf_list):
    image_path = os.path.join(non_leaf_dir, image)
    if i in train:
        image_dest_path = os.path.join(train_dir_non_leaf, image)
    elif i in valid:
        image_dest_path = os.path.join(valid_dir_non_leaf, image)
    elif i in valid:
        image_dest_path = os.path.join(test_dir_non_leaf, image)

    copyfile(image_path, image_dest_path)


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
            image_dest_path = os.path.join(train_dir_leaf, "train"+str(train_size)+".jpg")
            train_size += 1
        elif i in valid:
            image_dest_path = os.path.join(valid_dir_leaf, "valid"+str(valid_size)+".jpg")
            valid_size += 1
        else:
            image_dest_path = os.path.join(test_dir_leaf, "test"+str(test_size)+".jpg")
            test_size += 1
        
        copyfile(image_path, image_dest_path)
