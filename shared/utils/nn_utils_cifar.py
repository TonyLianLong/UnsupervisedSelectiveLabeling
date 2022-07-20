# Datasets
import random

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

from .config_utils import cfg
from .nn_utils import get_transform, normalization_kwargs_dict
from .augment import Augment, Cutout

# Credit: MoCov2 https://github.com/facebookresearch/moco/blob/main/moco/loader.py


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class PRETRAIN_CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {
            'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out


class PRETRAIN_CIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {
            'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out


def train_dataset_cifar(transform_name):
    if transform_name == "FixMatch-cifar10" or transform_name == "SCAN-cifar10" or transform_name == "FixMatch-cifar100" or transform_name == "SCAN-cifar100":
        normalization_kwargs = normalization_kwargs_dict[transform_name]
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            Augment(4),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs),
            Cutout(
                n_holes=1,
                length=16,
                random=True)])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs)])

        train_transforms = {
            'standard': transform_val, 'augment': transform_train}
    elif transform_name == "CLD-cifar10" or transform_name == "CLD-cifar100":
        # CLD uses MoCov2's aug: similar to SimCLR
        normalization_kwargs = normalization_kwargs_dict[transform_name]
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**normalization_kwargs)])
    else:
        raise ValueError(f"Unsupported transform type: {transform_name}")

    if cfg.DATASET.NAME == "cifar10":
        train_dataset_cifar = datasets.CIFAR10(
            root=cfg.DATASET.ROOT_DIR, train=True, transform=transform_train, download=True)

        val_dataset = datasets.CIFAR10(
            root=cfg.DATASET.ROOT_DIR, train=False, transform=transform_val, download=True)
    elif cfg.DATASET.NAME == "cifar100":
        train_dataset_cifar = datasets.CIFAR100(
            root=cfg.DATASET.ROOT_DIR, train=True, transform=transform_train, download=True)

        val_dataset = datasets.CIFAR100(
            root=cfg.DATASET.ROOT_DIR, train=False, transform=transform_val, download=True)

    return train_dataset_cifar, val_dataset


# Memory bank on CIFAR
def train_memory_cifar(root_dir, cifar100, transform_name, batch_size=128, workers=2, with_val=False):
    # Note that CLD uses the same normalization for CIFAR 10 and CIFAR 100

    transform_test = get_transform(transform_name)

    if cifar100:
        train_memory_dataset = datasets.CIFAR100(root=root_dir, train=True,
                                                 download=True, transform=transform_test)
        if with_val:
            val_memory_dataset = datasets.CIFAR100(root=root_dir, train=False,
                                                   download=True, transform=transform_test)
    else:
        train_memory_dataset = datasets.CIFAR10(root=root_dir, train=True,
                                                download=True, transform=transform_test)
        if with_val:
            val_memory_dataset = datasets.CIFAR10(root=root_dir, train=False,
                                                  download=True, transform=transform_test)

    train_memory_loader = torch.utils.data.DataLoader(
        train_memory_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False)

    if with_val:
        val_memory_loader = torch.utils.data.DataLoader(
            val_memory_dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True, drop_last=False)
        return train_memory_dataset, train_memory_loader, val_memory_dataset, val_memory_loader
    else:
        return train_memory_dataset, train_memory_loader
