#/usr/bin/python

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

class RNN_CNPI_BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        