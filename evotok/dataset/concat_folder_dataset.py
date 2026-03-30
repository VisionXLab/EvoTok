import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from .imagenet import SingleFolderDataset

def load_imagenet(path, ext="JPEG"):
    paths = glob(os.path.join(path, "*", f"*.{ext}"))
    print(f"load {len(paths)} images from imagenet.")
    return paths

def load_cc12m(path, ext="jpg"):
    paths = glob(os.path.join(path, "*", f"*.{ext}"))
    print(f"load {len(paths)} images from cc12m.")
    return paths

load_paths = {
    "cc12m": load_cc12m,
    "imagenet": load_imagenet,
}

class MultipleFolderDataset(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.image_paths = []

        for one_dir in directory.split('+'):
            info = one_dir.split(':')
            dataset_type, path = info[0], info[1]
            sample_num = (int(info[2])) if len(info) > 2 else None
            sub_image_paths = load_paths[dataset_type](path)
            if sample_num is not None:
                rng = random.Random(42)
                rng.shuffle(sub_image_paths)
                sub_image_paths = sub_image_paths[:sample_num]
                print(f"Sampled {len(sub_image_paths)} images from {dataset_type} dataset.")

            self.image_paths += sub_image_paths


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            print(f'{image_path} is not a valid image')
        return image, image_path
        # return image, torch.tensor(0)

def build_multiple_dataset(args, transform, split="train"):
    if split == "train":
        return MultipleFolderDataset(args.data_path, transform=transform)
    elif split == "val":
        return SingleFolderDataset(args.val_data_path, transform=transform)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    dataset = MultipleFolderDataset('cc12m:data/cc12m-wds/cc12m+imagenet:data/ImageNet-1K/train', transform=None)
    print(len(dataset))