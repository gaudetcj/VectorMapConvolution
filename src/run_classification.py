'''Train CIFAR10/100 with PyTorch.'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import numpy as np

from experiments.classification.models import resnet_standard, resnet_quaternion, resnet_vectormap
from models.optim.adamw import AdamW
from models.optim.cyclic_sched import CyclicLRWithRestarts
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--decays', nargs='+', help='epochs to decay lr at')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--batch', default=124, type=int, help='batch size')
parser.add_argument('--type', default='quat18', type=str, help='network type (real, quaternion, vector)')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    pytorch_dataset_class = torchvision.datasets.CIFAR10
    num_classes = 10
elif args.dataset == 'cifar100':
    pytorch_dataset_class = torchvision.datasets.CIFAR100
    num_classes = 100
else:
    raise ValueError('Unrecognized dataset name...')

trainset = pytorch_dataset_class(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, drop_last=True)

testset = pytorch_dataset_class(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, drop_last=True)


# Model
print('==> Building model..')
model_dict = {
    'real18': resnet_standard.ResNet18(num_classes),
    'quat18': resnet_quaternion.ResNet18(num_classes),
    'vect18': resnet_vectormap.ResNet18(num_classes),
    'real34': resnet_standard.ResNet34(num_classes),
    'quat34': resnet_quaternion.ResNet34(num_classes),
    'vect34': resnet_vectormap.ResNet34(num_classes),
    'real50': resnet_standard.ResNet50(num_classes),
    'quat50': resnet_quaternion.ResNet50(num_classes),
    'vect50': resnet_vectormap.ResNet50(num_classes),
}

net = model_dict.get(args.type, 'None')
if net == 'None':
    raise ValueError('Please choose network type in model_dict.')

net = net.to(device)
print('-'*50)
print('Param count:')
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
print('-'*50)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer = AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decays, gamma=0.1)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
scheduler = CyclicLRWithRestarts(optimizer, args.batch, len(trainset), restart_period=25, t_mult=2, policy="cosine", verbose=True)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    scheduler.step()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if 'quat' in args.type:
            inputs = torch.cat((torch.zeros((args.batch, 1, 32, 32)).to(device), inputs), dim=1)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.batch_step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if 'quat' in args.type:
                inputs = torch.cat((torch.zeros((args.batch, 1, 32, 32)).to(device), inputs), dim=1)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

    return [test_loss/(batch_idx+1), 100.*correct/total]


model_parameters = filter(lambda p: p.requires_grad, net.parameters())
param_count = sum([np.prod(p.size()) for p in model_parameters])
print('Param count: {}'.format(param_count))
test_stats = []
# for epoch in range(start_epoch, start_epoch+args.epochs):
#     train(epoch)
#     test_stats.append(test(epoch))

# with open("./results/results_{}_{}.csv".format(args.dataset, args.type), "w") as f:
#     wr = csv.writer(f)
#     wr.writerows(test_stats)

#torch.save(net, 'D:/Projects/VectorMapConvolution1/models/resnet18.pth')