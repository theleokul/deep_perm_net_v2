import os
from collections import defaultdict
from typing import List, Callable
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer


class PermutedMnistDataset(torch.utils.data.Dataset):
    def __init__(
        self
        , path: str='mnist'
        , download: bool=False
        , train: bool=True
        , transform: Callable=transforms.Compose([
            transforms.ToTensor()
            , transforms.Normalize((0.1307,), (0.3081,))
        ])
        , duplicates_multiplier: int=4
        , k: int=8
        , seed: int=42
    ):
        
        np.random.seed(seed)
        
        self.duplicates_multiplier = duplicates_multiplier
        self.k = k
        self.I = np.eye(k)
        
        ds = datasets.MNIST(
            path
            , download=True
            , train=True
            , transform=transform
        )
        
        self.binned_dataset = defaultdict(list)
        for x, y in ds:
            self.binned_dataset[y].append(x)
            
        self.min_len = min([len(digit_ds) for digit_ds in self.binned_dataset.values()])
            
    def __len__(self):
        return len(self.min_len * self.duplicates_multiplier)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, Number):
            idx = [idx]
                
        idx = [i // self.duplicates_multiplier for i in idx]
        
        X = []
        Y = []
        for i in idx:
            perm = np.random.permutation(np.arange(self.k, dtype=int))
            y = self.I[perm]
            
            x = []
            for pi in perm:
                x_i = self.binned_dataset[pi][i]
                x.append(x_i)
            
            X.append(x)
            Y.append(y)

        X = torch.tensor(X)
        Y = torch.tensor(Y)
            
        if len(X) == 1 and len(Y) == 1:
            X = X[0]
            Y = Y[0]
            
        return X, Y
