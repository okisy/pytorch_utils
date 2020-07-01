import torch
import torchvision
from torch import nn
from torchvision import models

import numpy as np
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    '''                    
    a= models.resnet50(pretrained=False)
    count = count_parameters(a)
    print (count)
    '''
    
def save_checkpoint(state, cp_file):
    torch.save(state, cp_file)

def set_seed(seed):    
    # set up seed
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                

