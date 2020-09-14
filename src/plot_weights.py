import os

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append(".")

from src.experiments.classification.models import resnet_vectormap


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

unnorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=6, shuffle=True)

net = resnet_vectormap.ResNet18(10).to(device)
net.load_state_dict(torch.load('C:/Projects/VectorMapConvolution_2/checkpoint/ckpt.t7')['net'])
#net = torch.load('models/resnet18.pth')

Ls = []
for nl in [net.conv1, net.layer1, net.layer2, net.layer3, net.layer4]:
    for p in nl.parameters():
        if p.shape == torch.Size([3, 3]):
            Ls.append(p.data.cpu().numpy())

Ls = sum([list(x.flatten()) for x in Ls], [])

plt.figure(figsize=(4, 3))
plt.hist(Ls, bins=np.linspace(-1.6, 1.6, 50))
plt.xlabel('L weight value')
plt.ylabel('Count')
plt.savefig(r"C:\Projects\VectorMapConvolution_2\data\cifar_data\hist.png", dpi=300, bbox_inches="tight")


# 
# for batch_idx, (inputs, targets) in enumerate(testloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = net.conv1(inputs)
#         break

# plt.figure(figsize=(20,20))
# t = 1
# for i in range(6):
#     img = unnorm(inputs[i])
#     img = img.data.cpu().numpy().transpose(1, 2, 0)
#     img = img / img.max()
#     img[img < 0] = 0
#     plt.subplot(6, 9, t)
#     plt.imshow(img)
#     plt.axis('off')
#     t += 1
#     for j in range(8):
#         c = outputs[i, 6*j:6*j+3, :, :].data.cpu().numpy().transpose(1, 2, 0)
#         c = c + np.abs(c.min())
#         c = c / c.max()
#         plt.subplot(6, 9, t)
#         plt.imshow(c)
#         plt.axis('off')
#         t += 1
# plt.show()

# mc = 0