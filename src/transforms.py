import random
import numpy as np
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    # if min_size < size
    if min_size < min(size):
        ow, oh = img.size
        # padh = size - oh if oh < size else 0
        # padw = size - ow if ow < size else 0
        padh = size[0] - oh if oh < size[0] else 0
        padw = size[1] - ow if ow < size[1] else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Resize:
    def __init__(self, size ,resize_target=True):
        self.size = size
        self.resize_target = resize_target

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        if self.resize_target:
            target = F.resize(target, self.size,interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomApply:
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, target):
        T.RandomApply(self.transforms, p=self.p)
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomPhotometricDistort:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, target):
        t = T.RandomPhotometricDistort(p=self.p)
        image = t(image)
        return image, target

class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    def __call__(self, image, target):
        crop_params = T.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        image = F.resized_crop(image, *crop_params, self.size)
        target = F.resized_crop(target, *crop_params, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomErase:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    def __call__(self, image, target):
        t = T.RandomErasing(p=self.p, scale=self.scale, ratio=self.ratio, value=self.value)
        image = t(image)
        return image, target


class RandomGrayScale:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, target):
        t = T.RandomGrayscale(p=self.p)
        image = t(image)
        return image, target


class RandomEqualize:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, image, target):
        t = T.RandomEqualize(p=self.p)
        image = t(image)
        return image, target

class GaussianBlur:
    def __init__(self, kernel_size=5, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
    def __call__(self, image, target):
        t = T.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
        image = t(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class PILToTensor:
    def __init__(self, target_numpy=False):
        self.target_numpy = target_numpy
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        if not self.target_numpy:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ToDtype:
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target):
        if not self.scale:
            return image.to(dtype=self.dtype), target
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
