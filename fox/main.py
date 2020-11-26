import torch
from .model import Model
from .config import MidasConfig, YoloV3Config
from torchsummary import summary

def something():
    net = Model(MidasConfig, YoloV3Config)
    
    if torch.cuda.is_available():
        net = net.cuda()

    print(summary(net, (3, 416, 416)))
