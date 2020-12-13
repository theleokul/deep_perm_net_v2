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


class SinkhornNormalizer(nn.Module):
    """
        Numerically stable version of making matrix to double stochastic
        https://www.groundai.com/project/learning-permutations-with-sinkhorn-policy-gradient/1#S4.SS2
    """

    def __init__(self, eps: float=1e-3, L: int=10, tau: float=0.05):
        super().__init__()
        self.eps = eps
        self.tau = tau
        self.L = L
            
    def forward(self, x):
        x /= self.tau

        for _ in range(self.L):
            # row normalization
            x = x - torch.logsumexp(x, dim=-2, keepdims=True)
            # column normalization
            x = x - torch.logsumexp(x, dim=-1, keepdims=True)
            
        # add a small offset ’eps’ to avoid numerical errors due to exp()
        return torch.exp(x) + self.eps


class SinkhornOptimizer_v1(nn.Module):
    def forward(self, X: torch.Tensor):
        return X

    
class SinkhornOptimizer_v2(nn.Module):
    def __init__(self, head_size: int, entropy_reg: float):
        e = np.ones((head_size, 1))
        Q = cp.Parameter((head_size, head_size))
        P_hat = cp.Variable((head_size, head_size))
        
        objective = cp.Minimize(
            cp.norm(P_hat - Q, p='fro') - entropy_reg * cp.sum(cp.entr(P_hat))
        )
        constraints = [
            P_hat @ e == e
            , e.T @ P_hat == e.T
            , P_hat >= 0
            , P_hat <= 1
        ]
        problem = cp.Problem(objective, constraints)
        
        self.model = CvxpyLayer(problem, parameters=[Q], variables=[P_hat])
        
    def forward(self, X: torch.Tensor):
        output, = self.model(X)
        return output
    
    
class SinkhornOptimizer_v3(nn.Module):
    def __init__(self, head_size: int, entropy_reg: float):
        # Generate data.
        e = np.ones((head_size, 1))
        M = cp.Parameter((head_size, head_size))
        P_hat = cp.Variable((head_size, head_size))

        objective = cp.Minimize(
            cp.norm(cp.multiply(P_hat, M), p='fro') - cp.sum(cp.entr(P_hat))
        )
        constraints = [
            P_hat @ e == e
            , e.T @ P_hat == e.T
            , P_hat >= 0
        ]
        problem = cp.Problem(objective, constraints)

        self.model = CvxpyLayer(problem, parameters=[M], variables=[P_hat])
        
    def forward(self, X: torch.Tensor):
        output, = self.model(X)
        return output
