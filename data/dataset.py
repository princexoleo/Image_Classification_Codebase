"""
author: Mazharul Islam Leon
created at: 2021-12-15

"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from data import get_train_transform, get_test_transform

import sys 
sys.path.append("..") 
import cfg

# from data import get_random_eraser


input_size = cfg.INPUT_SIZE
batch_size = cfg.BATCH_SIZE



class  SelfCustomDataset(Dataset):
    def __init__(self, label_file, imageset):
        '''
        img_dir:     
        '''
        
        with open(label_file, 'r') as f:
            #label_file， （label_file image_label)
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
      
      #   self.transforms=transform
        self.img_aug=True
        if imageset == 'train':
            self.transform= get_train_transform(size=cfg.INPUT_SIZE)
        else:
            self.transform = get_test_transform(size = cfg.INPUT_SIZE)
        self.input_size = cfg.INPUT_SIZE

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.img_aug:
            img =self.transform(img)


        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label)))
 
    def __len__(self):
        return len(self.imgs)





train_label_dir = cfg.TRAIN_LABEL_DIR
print(train_label_dir)
train_datasets = SelfCustomDataset(train_label_dir, imageset='train')
train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

val_label_dir = cfg.VAL_LABEL_DIR
val_datasets = SelfCustomDataset(val_label_dir, imageset='test')
val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=2)


if __name__ =="__main__":

    for images, labels in train_dataloader:
        print(labels)