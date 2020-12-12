import os
from collections import defaultdict
from typing import List, Callable
from numbers import Number
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import cvxpy as cp
from cvxpylayers.torch.cvxpylayer import CvxpyLayer
import pytorch_lightning as pl

import sinkhorn
import utils
import _deep_perm_net_v2 as deep_perm_net_v2
import permuted_mnist_dataset as pmnist
import lit_mnist_classifier


class LitDeepPermNet_v2(deep_perm_net_v2.DeepPermNet_v2, pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat.flatten(1), y.flatten(1))
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        losses = torch.as_tensor([o['loss'] for o in outputs])
        self.log('avg_train_loss', losses.mean(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.l1_loss(y_hat.flatten(1), y.flatten(1))

        output = {
            'val_loss': loss
        }
        self.log_dict(output, prog_bar=True)

        return output

    def validation_epoch_end(self, outputs):
        losses = []

        for o in outputs:
            losses.append(o['val_loss'])

        self.log_dict({
            'avg_val_loss': torch.as_tensor(losses).mean()
        }, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.003)

    def train_dataloader(self):
        train_dataset = pmnist.PermutedMnistDataset(
            'mnist'
            , download=True
            , train=True
            , transform=transforms.Compose([
                transforms.ToTensor()
                , transforms.Normalize((0.1307,), (0.3081,))
            ])
            , duplicates_multiplier=2
            , head_size=10
            , seed=42
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset
            , batch_size=8
            # , num_workers=1
            , shuffle=True
        )

        return train_loader

    def val_dataloader(self):
        val_dataset = pmnist.PermutedMnistDataset(
            'mnist'
            , download=True
            , train=False
            , transform=transforms.Compose([
                transforms.ToTensor()
                , transforms.Normalize((0.1307,), (0.3081,))
            ])
            , duplicates_multiplier=2
            , head_size=10
            , seed=42
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset
            , batch_size=8
            # , num_workers=1
        )

        return val_loader
        

if __name__ == "__main__":
    feature_extractor = lit_mnist_classifier.LitMNISTClassifier.load_from_checkpoint(
        checkpoint_path='logs_mnist_clf/version_1/checkpoints/epoch=14-avg_val_loss=0.0394-avg_val_acc=0.9876.ckpt'
    )
    model = LitDeepPermNet_v2(
        feature_extractor
        , feature_size=10
        , head_size=10
        , disable_feature_extractor_training=True
        , permutation_extractor_version='v1'
        , eps=1e-3
    )
    logger = pl.loggers.TensorBoardLogger(save_dir='./logs_deep_perm_net_v2', name='')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='avg_val_loss'
        , save_top_k=1
        , mode='min'
        , filepath=str(Path(logger.log_dir, 'checkpoints', '{epoch}-{avg_val_loss:.4f}-{avg_val_acc:.4f}'))
    )
    trainer = pl.Trainer(
        max_epochs=5
        , checkpoint_callback=checkpoint_callback
        , logger=logger
    )
    trainer.fit(model)
