from typing import Type
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT
import torchvision
import numpy as np
import os
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision
import scipy.misc as m
import imageio
import cv2

class pascalDataset(Dataset):
    def __init__(self, root_dir='../../../dataset/pascalVOC/VOCdevkit/VOC2012', split='train', labels=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.labels = labels

        self.image = os.path.join(self.root_dir, 'JPEGImages')
        self.split_data = os.path.join(self.root_dir, 'ImageSets/Main')
        self.seg_dic = os.path.join(self.root_dir, 'ImageSets/Segmentation')
        self.seg_data = []
        self.data = {}
        self.loadSplit() 

    def __len__(self):
        return len(self.data['img_id'])

    def __getitem__(self, idx):
        image, mask, label = self.loadImage(idx)
        image = FT.to_tensor(image)
        image = FT.resize(image, (224, 224))
        try:
            mask = FT.to_tensor(mask)
            # mask = FT.resize(mask, (224, 224), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            mask = FT.resize(mask, (224, 224), interpolation=0)
        except TypeError:
            pass
        # print(mask.shape)
        

        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask.float())
        label = torch.LongTensor([label])

        return image, mask, label
    
    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of masks, labels and additional_info
        """
        images = list()
        masks = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            masks.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, masks, labels

    def loadSplit(self):
        seg_dic = os.path.join(self.seg_dic, self.split + '.txt')
        with open(seg_dic, 'r') as txt:
            for line in txt:
                self.seg_data.append(line.split()[0])
        
        cat_order = 0
        for label in self.labels:
            dic = os.path.join(self.split_data, label + '_' + self.split + '.txt')
            with open(dic,'r') as txt:
                for line in txt:
                    if line.split()[1] == str(1): # -1: no object, 0: object but difficult
                        if self.split == 'train':
                            self.add_element(self.data ,'img_id', line.split()[0])
                            self.add_element(self.data, 'cat_id', cat_order)
                        else:
                            if line.split()[0] in self.seg_data:
                                self.add_element(self.data ,'img_id', line.split()[0])
                                self.add_element(self.data, 'cat_id', cat_order)
            cat_order += 1
        
    def loadImage(self, idx):
        label = self.data['cat_id'][idx]
        image_path = os.path.join(self.root_dir, 'JPEGImages', self.data['img_id'][idx] + '.jpg')
        mask_path = os.path.join(self.root_dir, 'SegmentationClass', self.data['img_id'][idx] + '.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # try:
        #     mask = cv2.imread(mask_path)
        #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #     mask = self.encode_segmap(mask, label)
        # except cv2.error:
        #     mask = torch.zeros(3,224,224)
        if self.split == 'train':
            mask = torch.zeros(1,224,224)
        else:
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = self.encode_segmap(mask, label)

        return image, mask, label

    def add_element(self, dict, key, value):
        if key not in dict:
            dict[key] = []
        dict[key].append(value)
    
    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )
    
    def encode_segmap(self, mask, labels):
        pascal_label = ['background','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', \
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii

        binary_mask = np.where(label_mask==pascal_label.index(self.labels[labels]), 1, 0)

        return binary_mask


    def convert_image_np(self, inp, image):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array((0.485, 0.456, 0.406))
        std = np.array((0.229, 0.224, 0.225))
        if image:
            # inp = std * inp + mean
            inp = np.clip(inp, 0., 1.)
        
        return inp

if __name__ == "__main__":
    labels = ['car']
    dataSet = pascalDataset(labels=labels, split='val')

    loader = torch.utils.data.DataLoader(dataSet,
                                        batch_size=16, 
                                        shuffle=True,
                                        num_workers=0,
                                        drop_last=True,
                                        collate_fn=dataSet.collate_fn,
                                        )

    for i, (image, mask, label) in enumerate(loader):
        in_grid = dataSet.convert_image_np(
                torchvision.utils.make_grid(image, nrow=4), True)
        
        mask_grid = dataSet.convert_image_np(
                torchvision.utils.make_grid(mask, nrow=4), False)

        plt.close('all')
        fig = plt.figure(figsize=(12,12))
        fig.tight_layout()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        ax1.imshow(in_grid)
        ax1.set_title('input')

        ax2.imshow(mask_grid, cmap='gray')
        ax2.set_title('mask')

        fig.tight_layout()

        fig.savefig('./exampleImage/' + 'pascaltest2' + '.png')
        exit(1) 