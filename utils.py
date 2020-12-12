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


class Reshape(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape
        
    def forward(self, X: torch.Tensor):
        batch_size = X.shape[0]
        X = X.reshape(batch_size, *self.shape)
        return X


class MultiHeadFeatureExtractor(nn.Module):
    def __init__(self, feature_extractor: nn.Module, feature_size: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_size = feature_size
        
    def forward(self, X: torch.Tensor):
        batch_size, head_size, channel_size, height, width = X.shape
        X = X.reshape(batch_size * head_size, channel_size, height, width)

        output = self.feature_extractor(X)
        output = output.reshape(batch_size, head_size * self.feature_size)

        return output
