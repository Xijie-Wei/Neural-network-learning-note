# Neural-network-learning-note

## Introduction
This is a note taken down when I was learning to set up and train a network with Pytorch.
This project is about getting familiar with Pytorch with the tutorial 'Deep Learning with PyTorch: A 60 Minute Blitz' by Soumith Chintala from: 'https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html' And try to set up a hand-writing number classification neural network with dataset MNIST. This repository is a note and Ill try to explain my code as possible. However, parts like the neural network structure are what Im not familiar with (which I still need to spend time learning, Im just a beginner in this field), so I`ll just skip them.

## Code 
In this part, Ill try to explain my code as much as possible.<br/>
The libs used in this project are `torch` and `torchvision`
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
```
Then we declare a class `net` which contains the structure of the neural network. Initially, following the tutorial, I declared two Conv2d layers and 3 linear layers. 
```
self.conv1 = nn.Conv2d(1,6,5)
self.conv2 = nn.Conv2d(6,16,5)
self.fc1 = nn.Linear(16*4*4,120)
self.fc2 = nn.Linear(120,84)
self.fc3 = nn.Linear(84,10)
```
Im not familiar with Conv2d layers so Ill just skip them. But one important thing is that the size of the output of a layer should be equal to the size of the next layer, i.e. for `fc1` the output is set as 120, and the input of `fc2` is 120. This should be true for all layers, may need to add function like `torch.flatten`.

Then we input the MNIST dataset
```
# Load MNIST
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
     ])

batch_size = 1

train_set = torchvision.datasets.MNIST(root='./data',train = True,
                                        download = True, transform = transform)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
print("MNIST loaded")
```
The `transform` is declared with two purpose, one is to put the data in `tensor` and normalize it, for details, check `https://pytorch.org/vision/0.9/_modules/torchvision/transforms/transforms.html#Compose`
