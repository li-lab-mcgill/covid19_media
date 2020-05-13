from torch import nn
import torch

class TrainableEltwiseLayer(nn.Module):
    def __init__(self, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones(h, w))  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size h-w
        return x * self.weights  # element-wise multiplication


