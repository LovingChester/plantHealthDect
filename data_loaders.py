import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset, IterableDataset, DataLoader
import time
from tqdm import tqdm

class LeafDataset(IterableDataset):

    def __init__(self, data_dir, num_workers):
        # data_dir: directory containing leaf data
        # number of workers used to load data
        self.data_dir = data_dir

        if num_workers <= 0:
           self.num_workers = 1
        else:
           self.num_workers = num_workers

        # transform PIL image to tensor
        self.to_tensor = transforms.ToTensor()

        # transform images that are not 256x256
        self.resize = transforms.Resize((256, 256))

    def process_data(self, data_dir, worker_id):
        count = 0
        files = os.listdir(data_dir)
        for idx, file in enumerate(files):
            file_path = os.path.join(data_dir, file)
            image_list = os.listdir(file_path)
            for image in image_list:
                image_path = os.path.join(file_path, image)
                # converts the image to tensor
                image_data = self.to_tensor(Image.open(image_path))

                if image_data.shape[0] == 4: continue

                if image_data.shape[1] != 256 or image_data.shape[2] != 256: 
                    image_data = self.resize(image_data)

                if count % self.num_workers == worker_id:
                    yield image_data, torch.tensor(idx)
                
                count += 1
    
    def __len__(self):
        length = 0
        files = os.listdir(self.data_dir)
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            image_list = os.listdir(file_path)
            length += len(image_list)

        return length
    
    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
           worker_id = worker.id
           num_workers = worker.num_workers
        else:
           worker_id = 0
           num_workers = 1

        for idx, (image_data, index) in enumerate(self.process_data(self.data_dir, worker_id)):
            yield image_data, index

        #return self.process_data(self.data_dir)


if __name__ == "__main__":

    num_workers = 0
    leaf_dataset = LeafDataset(data_dir="../Plant_leave_diseases_dataset_with_augmentation", num_workers=num_workers)
    loader = DataLoader(leaf_dataset, batch_size=10, num_workers=num_workers)

    start = time.time()
    for idx, (data, index) in enumerate(tqdm(loader)):

        # print(index.shape)
        # print(data.shape)
        pass
    
    end = time.time()

    print("time spent:", end-start)

