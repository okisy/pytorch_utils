import torch
import torchvision
from torch import nn
from torchvision import models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
a= models.resnet50(pretrained=False)
count = count_parameters(a)
print (count)
