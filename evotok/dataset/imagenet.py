import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from .coco import SingleFolderDataset

def build_imagenet(args, transform, split="train"):
    if split == "train":
        return ImageFolder(args.data_path, transform=transform)
    elif split == "val":
        return SingleFolderDataset(args.val_data_path, transform=transform)
    else:
        raise NotImplementedError