import torch
from midas.midas.midas_net import MidasNet

net = MidasNet("../weights/midas.pt")

features, out = net(torch.randn(1, 3, 64, 64))
print(features.shape)
