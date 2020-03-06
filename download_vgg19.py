import torch 
import torchvision
import os

os.environ['TORCH_HOME'] = 'data/VGG' 
torchvision.models.vgg19(pretrained=True, progress=True)
