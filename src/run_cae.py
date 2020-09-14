import torch
from torch import nn
from experiments.cae.models.cae_models import CAE, QCAE, VCAE
from models.optim.adamw import AdamW

import os
from imageio import imread,imwrite
import numpy as np
import sys


def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


MODEL         = 'QCAE'
NUM_EPOCHS    = 3001
LEARNING_RATE = 0.01

if MODEL == 'QCAE':
    net  = QCAE()
elif MODEL == 'VCAE':
    net = VCAE()
else:
    net  = CAE()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

#
# LOAD PICTURE
#

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

train = rgb2gray(imread('data/cae_data/kodim05.png'))
imwrite("data/cae_data/out/save_image_gray_training.png", train)
imwrite("data/cae_data/out/girl_image_gray_training.png", rgb2gray(imread('data/cae_data/kodim04.png')))
imwrite("data/cae_data/out/parrot_image_gray_training.png",rgb2gray(imread('data/cae_data/kodim23.png')))
train = train / 255.

test = imread('data/cae_data/kodim04.png')
test = test / 255.

nb_param = sum(p.numel() for p in net.parameters() if p.requires_grad)

print("Model Info --------------------")
print("Number of trainable parameters : "+str(nb_param))

if MODEL == 'QCAE' or MODEL == 'VCAE':
    npad  = ((0, 0), (0, 0), (1, 0))
    train = np.pad(train, pad_width=npad, mode='constant', constant_values=0)
    train = np.transpose(train, (2,0,1))
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))


    test = np.pad(test, pad_width=npad, mode='constant', constant_values=0)
    test = np.transpose(test, (2,0,1))
    test = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

else:
    train = np.transpose(train, (2,0,1))
    train = np.reshape(train, (1, train.shape[0], train.shape[1], train.shape[2]))

    test  = np.transpose(test, (2,0,1))
    test  = np.reshape(test, (1, test.shape[0], test.shape[1], test.shape[2]))

train = torch.from_numpy(train).float().to(device)
test  = torch.from_numpy(test).float().to(device)


for epoch in range(NUM_EPOCHS):

    output = net(train)
    loss   = criterion(output, train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("epoch "+str(epoch+1)+", loss_train "+str(loss.cpu().item()))

    if (epoch %100) == 0:

        output = net(test)
        out    = output.cpu().data.numpy()
        if MODEL == 'QCAE' or MODEL == 'VCAE':
            out = np.transpose(out, (0,2,3,1))[:,:,:,1:]
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))
        else:
            out = np.transpose(out, (0,2,3,1))
            out = np.reshape(out, (out.shape[1], out.shape[2], out.shape[3]))


        imwrite("data/cae_data/out/save_image"+str(epoch)+".png", out)