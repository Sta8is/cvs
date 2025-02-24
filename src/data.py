from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.v2 as Tv2
from .transforms import *
import torch
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import copy


class CoreValuesLabeled(Dataset):
    """Dataset class for labeled geological core images"""
    
    def __init__(self, 
                 image_files: List[str], 
                 split: str = 'train',
                 image_size: Tuple[int, int] = (1344, 364)):
        """
        Initialize dataset for labeled images.
        
        Args:
            image_files: List of image file paths
            split: Dataset split ('train' or 'val')
            image_size: Input image dimensions (height, width)
        """
        self.image_files = image_files
        self.split = split
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.img_size = image_size

        
        self.transform_train = Compose([
            Resize(self.img_size, resize_target=True),
            RandomHorizontalFlip(flip_prob=0.5),
            RandomPhotometricDistort(p=0.5),
            PILToTensor(),
            ToDtype(torch.float, scale=True),
            Normalize(mean=self.mean, std=self.std)
        ])

        self.transform_val = Compose([
            Resize(self.img_size, resize_target=True),
            PILToTensor(),
            ToDtype(torch.float, scale=True),
            Normalize(mean=self.mean, std=self.std)
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (image, label) tensors
        """
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        label = np.load(image_path.replace('img.png', 'mask_gt.npy'))
        label = torch.from_numpy(label).unsqueeze(0)

        transform = self.transform_train if self.split == 'train' else self.transform_val
        return transform(image, label)

def obtain_cutmix_box(
    img_size: Tuple[int, int],
    p: float = 0.5,
    size_min: float = 0.02,
    size_max: float = 0.4,
    ratio_1: float = 0.3,
    ratio_2: float = 1/0.3
) -> torch.Tensor:
    """
    Generate a random cutmix mask.
    
    Args:
        img_size: Image dimensions (height, width)
        p: Probability of applying cutmix
        size_min: Minimum size ratio
        size_max: Maximum size ratio
        ratio_1: Minimum aspect ratio
        ratio_2: Maximum aspect ratio
        
    Returns:
        Binary mask tensor
    """
    h, w = img_size
    mask = torch.zeros(h, w)
    
    if np.random.random() > p:
        return mask

    while True:
        area = np.random.uniform(size_min, size_max) * (h * w)
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(area / ratio))
        cutmix_h = int(np.sqrt(area * ratio))
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        
        if x + cutmix_w <= w and y + cutmix_h <= h:
            mask[y:y + cutmix_h, x:x + cutmix_w] = 1
            break

    return mask


# class CoreValuesSemi(Dataset):
#     """Dataset class for semi-supervised learning with both labeled and unlabeled images"""
    
#     def __init__(self, 
#                  image_files: List[str], 
#                  split: str = 'train_labeled',
#                  image_size: Tuple[int, int] = (1344, 364)):
#         """
#         Initialize semi-supervised dataset.
        
#         Args:
#             image_files: List of image file paths
#             split: Dataset split ('train_labeled', 'train_unlabeled', or 'val')
#             image_size: Input image dimensions (height, width)
#         """
#         self.image_files = image_files
#         self.split = split
#         self.mean = [0.485, 0.456, 0.406]
#         self.std = [0.229, 0.224, 0.225]
#         self.img_size = image_size

#         # Base transforms
#         self.transform_train = T.Compose([
#             T.RandomResizedCrop(
#                 size=self.img_size, 
#                 scale=(0.5, 2.0),
#                 ratio=(0.1, 6)
#             ),
#             RandomHorizontalFlip(flip_prob=0.5),
#             RandomPhotometricDistort(p=0.5),
#             PILToTensor(),
#             ToDtype(torch.float, scale=True),
#             Normalize(mean=self.mean, std=self.std)
#         ])

#         self.transform_val = T.Compose([
#             Resize(self.img_size, resize_target=True),
#             PILToTensor(),
#             ToDtype(torch.float, scale=True),
#             Normalize(mean=self.mean, std=self.std)
#         ])

#         # Additional transforms for unlabeled data
#         self.transform_unlabeled = T.Compose([
#             RandomGrayScale(p=0.2),
#             RandomEqualize(p=0.2),
#             RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5)
#         ])

#     def __len__(self) -> int:
#         return len(self.image_files)

#     def __getitem__(self, idx: int) -> Tuple:
#         """
#         Get a single item from the dataset.
        
#         Args:
#             idx: Index of the item
            
#         Returns:
#             Tuple containing transformed images and labels based on split type
#         """
#         image_path = self.image_files[idx]
#         image = Image.open(image_path).convert('RGB')
        
#         # Handle labeled data
#         if self.split in ['train_labeled', 'val']:
#             label = np.load(image_path.replace('img.png', 'mask_gt.npy'))
#             label = torch.from_numpy(label).unsqueeze(0)
#         else:
#             label = torch.zeros(1, *self.config.img_size)

#         # Validation split
#         if self.split == 'val':
#             return self.transform_val(image, label)

#         # Apply base transforms
#         image, label = self.transform_train(image, label)

#         # Labeled training data
#         if self.split == 'train_labeled':
#             return image, label

#         # Unlabeled training data
#         image_weak = image.clone()
#         image_strong1 = image.clone()
#         image_strong2 = image.clone()
        
#         image_strong1, _ = self.transform_unlabeled(image_strong1, label)
#         image_strong2, _ = self.transform_unlabeled(image_strong2, label)
        
#         cutmix_box1 = obtain_cutmix_box(self.config.img_size, p=0.5)
#         cutmix_box2 = obtain_cutmix_box(self.config.img_size, p=0.5)

#         return image_weak, image_strong1, image_strong2, cutmix_box1, cutmix_box2

class CoreValuesSemi(Dataset):
    def __init__(self, image_files, split='train_labeled', image_size=(1344, 364)):
        self.image_files = image_files
        self.split = split
        self.img_size = image_size

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.transform_train = []
        #Random crop 50-100% of original size # Allow some aspect ratio variation
        self.transform_train += [RandomResizedCrop(size=self.img_size, scale=(0.5, 2.0),ratio=(0.1, 6))]
        # self.transform_train += [Resize(self.img_size)]
        # self.transform_train += [RandomVerticalFlip(flip_prob=0.5)]
        # self.transform_train += [RandomErase(p=0.2, scale=(0.02, 0.15))]
        # self.transform_train += [RandomGrayScale(p=0.5)]
        self.transform_train += [RandomHorizontalFlip(flip_prob=0.5)]
        self.transform_train += [RandomPhotometricDistort(p=0.5)]
        self.transform_train += [PILToTensor()]
        self.transform_train += [ToDtype(torch.float, scale=True)]
        # self.transform_train += [GaussianNoise(m=0.0, std=0.1)]
        self.transform_train += [Normalize(mean=self.mean, std=self.std)]
        self.transform_train = Compose(self.transform_train)

        self.transform_val = []
        self.transform_val += [Resize(self.img_size, resize_target=True)]
        self.transform_val += [PILToTensor()]
        self.transform_val += [ToDtype(torch.float, scale=True)]
        self.transform_val += [Normalize(mean=self.mean, std=self.std)]
        self.transform_val = Compose(self.transform_val)

        self.transform_unlabaled = []
        self.transform_unlabaled += [RandomGrayScale(p=0.2)]
        self.transform_unlabaled += [RandomEqualize(p=0.2)]
        self.transform_unlabaled += [Tv2.RandomApply([GaussianBlur(kernel_size=5)], p=0.5)]
        self.transform_unlabaled = Compose(self.transform_unlabaled)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        if self.split == 'train_labeled' or self.split == 'val':
            label = np.load(image_path.replace('img.png', 'mask_gt.npy'))
            label = torch.from_numpy(label).unsqueeze(0)
        else:
            label = torch.zeros(1, *self.img_size)
        if self.split == 'val':
            return self.transform_val(image, label)
    
        image, label = self.transform_train(image, label)

        if self.split == 'train_labeled':
            return image, label

        image_weak, image_strong1, image_strong2 = copy.deepcopy(image), copy.deepcopy(image), copy.deepcopy(image)
    
        image_strong1, _ = self.transform_unlabaled(image_strong1, label)
        cutmix_box1 = obtain_cutmix_box(self.img_size, p=0.5)
        image_strong2, _ = self.transform_unlabaled(image_strong2, label)
        cutmix_box2 = obtain_cutmix_box(self.img_size, p=0.5)
        return image_weak, image_strong1, image_strong2, cutmix_box1, cutmix_box2