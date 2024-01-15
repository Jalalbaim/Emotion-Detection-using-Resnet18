import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(AttentionBlock, self).__init__()
        self.key = nn.Linear(in_channels, out_channels)
        self.Query = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.fc = nn.Linear(out_channels, num_classes)
    
    def forward(self, x):
        key = self.key(x)
        Query = self.Query(x)
        value = self.value(x)
        attention = torch.matmul(Query, key.transpose(-2, -1))
        attention = attention / np.sqrt(key.size(-1))
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, value)
        out = self.fc(out)
        return out
