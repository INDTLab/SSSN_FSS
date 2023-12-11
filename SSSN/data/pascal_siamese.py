r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import heapq,cv2
import random

import albumentations as A
import torch.nn as nn

class DatasetPASCAL_siamese(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
       img1_name, img2_name, class_id = self.sample_episode(idx)
      
       img1 = self.read_img(img1_name)
       img2 = self.read_img(img2_name)
       img1 = self.transform(img1)
       img2 = self.transform(img2)

       batch = {'img_1': img1, 'img_2': img2, 'target':class_id}
       return batch 

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        img1_name, class_id = self.img_metadata[idx]
        
        if idx % 2 == 0:
            while True:
                img2_name = np.random.choice(self.img_metadata_classwise[class_id], 1, replace=False)[0]
                if img1_name != img2_name:
                    break
            target = torch.tensor(1, dtype=torch.float)
        else:
            if self.split == 'trn':
                for fold_id in range(self.nfolds):
                    if fold_id == self.fold:
                        continue
            
            other_selected_class = random.randint(0, 19)
            
            while other_selected_class == class_id or other_selected_class in range(self.fold*5,(self.fold+1)*5): #增加避免 val
                other_selected_class = random.randint(0, 19)
        
            img2_name = np.random.choice(self.img_metadata_classwise[other_selected_class], 1, replace=False)[0]
            target = torch.tensor(0, dtype=torch.float)
            
        return img1_name, img2_name, target

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):
        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('../../Datasets_ASSN/VOC2012/splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))
        
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
