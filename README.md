# Neural-network-learning-note

## Introduction
This is a note taken down when I was learning to set up and train a network with Pytorch.
This project is about getting familiar with Pytorch with the tutorial 'Deep Learning with PyTorch: A 60 Minute Blitz' by Soumith Chintala from: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html And try to set up a hand-writing number classification neural network with dataset MNIST. This repository is a note and Ill try to explain my code as possible. However, parts like the neural network structure are what Im not familiar with (which I still need to spend time learning, Im just a beginner in this field), so I`ll just skip them.

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
The `transform` is declared with two purposes, one is to put the data in `tensor` and normalize it, for details, check https://pytorch.org/vision/0.9/_modules/torchvision/transforms/transforms.html#Compose for details. The `batch_size` should be the channel size? for MNIST database, the image is a grayscale image, so only 1 is set, but for other datasets with more color, it should be set higher like 3, and this should fit the size of the input in the first layer of the network (Im not sure about this).

Then we need to define the loss function the loss function is a built-in function in optim, which uses SGD. There are two main factors loss rate `lr` and `momentum`, which would influence the quality of the training, as why they are set 0.001 and 0.9, emm... IDK.
```
# Define loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(),lr=0.001,momentum = 0.9)
```

Then we go to the main part of training the network
```
for epoch in range(3): #number of epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Load data list
        inputs,labels = data[0].to('cuda'),data[1].to('cuda')
        # Zero the gradient
        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i +1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
```
epoch means how many times we use the dataset to train the network. Then we traverse `data` in `trainloader`, the `data` contains two parts `inputs` and `labels` where `inputs` are image data and `labels` are expected output. Then call `optimizer.zero_grad()` to set the gradient of the parameters 0. After that, input the `inputs` into the network and calculate `loss` using `criterion(outputs,labels)`, then call `loss.backward()` and `optimizer.step()`.
After training the network, call
```
PATH = './cifar_net.pth'
torch.save(network.state_dict(), PATH)
```
To store the parameter information.
## Result
After training the network, I write a `test.py` and find out the accuracy is 98.52% when testing with test dataset in MNIST
![图片](https://github.com/user-attachments/assets/54b34e90-7156-4824-b0a1-263693f44b77)

## Note
This note is based on my limited knowledge. As Im a beginner in training neural networks, there are probably mistakes in the note.
