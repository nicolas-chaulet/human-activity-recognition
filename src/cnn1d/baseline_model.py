"""
This file defines the core research contribution   
"""
import os
import sys
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pytorch_lightning as pl
import math
from sklearn.metrics import confusion_matrix
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "../..")
UCI_DATA_PATH = os.path.join(ROOT,'data/UCI HAR Dataset')

from src.dataset.ucihar import UCIHAR


class HARCNN(pl.LightningModule):

    def __init__(self, hparams):
        super(HARCNN, self).__init__()
        self.test_dataset = UCIHAR(UCI_DATA_PATH, split='test')
        self.train_dataset = UCIHAR(UCI_DATA_PATH, split='train')

        print("Size of train dataset= %i" % len(self.train_dataset))
        print("Size of test dataset= %i" % len(self.test_dataset))

        n_features = self.train_dataset.n_features
        n_timesteps = self.train_dataset.n_timesteps
        n_labels = self.train_dataset.n_labels

        mid_feature_size = 64
        mid_mlp = 100
        time_kernel_size = 3
        dropout = 0.5
        pool_size = 2
        out_maxpool = 62

        # not the best model...
        self.hparams = hparams
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(n_features, mid_feature_size, time_kernel_size),
            torch.nn.ReLU(),
            torch.nn.Conv1d(mid_feature_size, mid_feature_size, time_kernel_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.MaxPool1d(pool_size)
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(out_maxpool * mid_feature_size, mid_mlp),
            torch.nn.ReLU(),
            torch.nn.Linear(mid_mlp, n_labels)
        )
        print(self)

    def forward(self, x):
        feats = self.cnn(x)
        feats = feats.reshape(x.shape[0], -1)
        last_layer = self.mlp(feats)
        return  F.log_softmax(last_layer, dim = -1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        tqdm_dict = {'loss':  F.nll_loss(logits, y), 'acc':  accuracy(logits, y)}
        return {'loss':tqdm_dict['loss'],'log':tqdm_dict}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        conf = get_confusion(logits, y, np.arange(0,self.train_dataset.n_labels))
        return {'val_loss':F.nll_loss(logits, y), 'val_acc': accuracy(logits, y), 'val_conf': conf}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        confs  = None
        for x in outputs:
            if confs is None:
                confs = x['val_conf']
            else:
                confs += x['val_conf']
        
        tqdm_dict = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print(tqdm_dict)
        print(confs)
        return {'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"], 'val_conf': confs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--batch_size', default=32, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=10, type=int)

        return parser

def accuracy(output, y):
    pred = output.max(-1)[1]
    return torch.sum(pred == y).float() / y.shape[0]

def get_confusion(output, y, labels):
    pred = output.max(-1)[1]
    return confusion_matrix(y,pred, labels)