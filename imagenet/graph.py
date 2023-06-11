# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:41:42 2023

@author: Dido
"""
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

model = models.__dict__['resnet18']()
model.fc=nn.Linear(512,200,True)
print(model)
train_sampler = None

train_dataset = datasets.FakeData(100000, (3, 64, 64), 200, transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler)
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
dataiter = iter(train_loader)
images, labels = next(dataiter)
writer.add_graph(model, images)