import glob
import json
import math
import os
import pickle
import random
import warnings
from hashlib import md5
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data as data
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())

# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def calc_dataset_md5(dataset):
    try:
        md5_val =  md5(json.dumps(dataset.__dict__, sort_keys=True).encode('utf-8')).hexdigest()
    except Exception as e:
        print(f'Failed to calculate md5 for dataset: {e}. Using pickle instead.')
        data_bytes = pickle.dumps(dataset)
        md5_val = md5(data_bytes).hexdigest()
    return md5_val

class FeatureStats:

    def __init__(
        self,
        capture_all=False,
        capture_mean_cov=False,
        max_items=None,
        only_stats_mode=False,
        loaded_mean=None,
        loaded_cov=None,
    ):
        self.only_stats_mode = only_stats_mode
        if only_stats_mode:
            # load pre-computed mean and cov
            assert loaded_mean is not None and loaded_cov is not None, 'loaded_mean and loaded_cov must be provided in only_stats_mode'
            self.loaded_mean = loaded_mean
            self.loaded_cov = loaded_cov
        else:
            assert loaded_mean is None and loaded_cov is None, 'loaded_mean and loaded_cov must be None if only_stats_mode is False'
            self.loaded_mean = self.loaded_cov = None
            self.capture_all = capture_all
            self.capture_mean_cov = capture_mean_cov
            self.max_items = max_items
            self.num_items = 0
            self.num_features = None
            self.all_features = None
            self.raw_mean = None
            self.raw_cov = None

    def set_num_features(self, num_features):
        if self.only_stats_mode:
            raise ValueError('Cannot set num_features in only_stats_mode')

        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        if self.only_stats_mode:
            return True
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        if self.only_stats_mode:
            raise ValueError('Cannot append in only_stats_mode')

        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1):
        if self.only_stats_mode:
            raise ValueError('Cannot append in only_stats_mode')

        assert isinstance(x, torch.Tensor) and x.ndim == 2
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.float().cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        if self.only_stats_mode:
            return self.loaded_mean, self.loaded_cov
        else:
            if self.capture_mean_cov:
                mean = self.raw_mean / self.num_items
                cov = self.raw_cov / self.num_items
                cov = cov - np.outer(mean, mean)
                
            elif self.capture_all:
                features = self.get_all()
                mean = np.mean(features, axis=0)
                cov = np.cov(features, rowvar=False)

            else:
                raise ValueError('No stats captured')
            
            return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(stats_path):
        stats_path = Path(stats_path)
        if stats_path.suffix == '.pkl': # pickle file, load as FeatureStats
            with open(stats_path, 'rb') as f:
                s = edict(pickle.load(f))
            obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
            obj.__dict__.update(s)
        elif stats_path.suffix == '.npz': # npz file, ADM's precomputed mean and cov
            data = np.load(stats_path)
            obj = FeatureStats(only_stats_mode=True, loaded_mean=data['mu'], loaded_cov=data['sigma'])
        else:
            raise ValueError(f'Unknown file extension: {stats_path}')
        return obj
    

    def __add__(self, other):
        if not isinstance(other, FeatureStats):
            return NotImplemented

        if self.only_stats_mode or other.only_stats_mode:
            raise ValueError('Cannot add FeatureStats with only_stats_mode=True')

        if self.num_features != other.num_features:
            raise ValueError('Cannot add FeatureStats with different num_features')

        # Check compatibility of capture_all and capture_mean_cov
        if self.capture_all != other.capture_all:
            raise ValueError('Cannot add FeatureStats with different capture_all settings')

        if self.capture_mean_cov != other.capture_mean_cov:
            raise ValueError('Cannot add FeatureStats with different capture_mean_cov settings')

        # Create a new FeatureStats instance
        result = FeatureStats(
            capture_all=self.capture_all,
            capture_mean_cov=self.capture_mean_cov,
            max_items=None,  # No limit for the merged result
            only_stats_mode=False
        )
        result.num_features = self.num_features
        result.num_items = self.num_items + other.num_items

        # Combine all_features if capture_all is True
        if self.capture_all:
            result.all_features = self.all_features + other.all_features
        else:
            result.all_features = None

        # Combine raw_mean and raw_cov if capture_mean_cov is True
        if self.capture_mean_cov:
            result.raw_mean = self.raw_mean + other.raw_mean
            result.raw_cov = self.raw_cov + other.raw_cov
        else:
            result.raw_mean = None
            result.raw_cov = None

        return result

