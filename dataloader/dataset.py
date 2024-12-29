from __future__ import print_function

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

import os
import pickle
import cv2
from dataloader.classNamesAndTemplates import imagenet_classes, imagenet_templates, imagenet_offset2class_dict, FC100_offset2class_dict,CIFAR_FS_offset2class_dict

# =============== smkd dataset =================
class SMKD_dataset(datasets.ImageFolder):
    def __init__(self, dataset, backbone_type='SMKD', split="train"):
        datapath_dict = {
        'miniImageNet':{
            'data_path': 'data/miniImageNet'
            },  
        'CIFAR_FS':{
            'data_path': 'data/cifar_fs/'
            },
        'FC100':{
            'data_path': 'data/FC100/'
            },
        'tieredImageNet':{
            'data_path': 'data/tiered-imagenet-tools/tiered_imagenet'
            },
        }
        self.dataset = dataset
        self.split = split
        assert backbone_type in ['SMKD', 'Res12'], "backbone_type:{} should be in ['SMKD', 'Res12']".format(backbone_type)
        assert split in ['train', 'val', 'test'], "split:{} should be in ['train', 'val', 'test']".format(split)
        print("Current train dataset: ", self.dataset, self.split)
        
        if 'mini' in self.dataset or 'tiered' in self.dataset:
            mean = tuple([0.485, 0.456, 0.406])
            std = tuple([0.229, 0.224, 0.225])
            if backbone_type == 'SMKD':
                img_size = 360   
                img_resize = 320
            elif backbone_type == 'Res12':
                img_size = 92   
                img_resize = 84
        elif 'CIFAR_FS' in self.dataset or 'FC100' in self.dataset:
            mean = tuple([x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]])
            std = tuple([x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]])  
            if backbone_type == 'SMKD':
                img_size = 256
                img_resize = 224
            elif backbone_type == 'Res12':
                img_size = 92   
                img_resize = 84
        val_transform = transforms.Compose([
            transforms.Resize(int(img_size), interpolation=3),
            transforms.CenterCrop(int(img_resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_transform = transforms.Compose([
            transforms.Resize(int(img_size), interpolation=3),
            transforms.RandomCrop(img_resize),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if split in ['val', 'test']:
            transform = val_transform
        else:
            transform = train_transform
        super(SMKD_dataset, self).__init__(root=os.path.join(datapath_dict[dataset]['data_path'], split), transform=transform)
        self.offset2label = self.class_to_idx
        dataset_type = 'imagenet' if self.dataset in ['miniImageNet', 'tieredImageNet'] else self.dataset
        offset2class_dict = eval(f"{dataset_type}_offset2class_dict")
        self.label2class_name = {label:offset2class_dict[off] for off, label in self.offset2label.items() }
        
        self.classes = list(self.label2class_name.values())
        self.label2classid = dict(zip(list(range(len(self.classes))), list(range(len(self.classes))))) 
        self.labels = [label for path, label in self.imgs]
        
        label2class_path = "dataloader/label2class/{}/{}_label2class.pkl".format(self.dataset, self.split)
        if os.path.isfile(label2class_path):
            with open(label2class_path, 'rb') as f:
                self.label2class = pickle.load(f)


