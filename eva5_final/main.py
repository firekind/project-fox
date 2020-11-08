import torch
from midas.midas.midas_net import MidasNet

net = MidasNet("../weights/midas.pt")

l2, l3, l4, out = net(torch.randn(1, 3, 416, 416))
print(l2.shape)
print(l3.shape)
print(l4.shape)
