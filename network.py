# This program is about to train a network with MNIST database

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# Declare a network
class net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*4*4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,inputs):
        """
            This network structure is based on
            Deep Learning with PyTorch: A 60 Minute Blitz
            By Soumith Chintala
            From https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
        """
        c1 = F.relu(self.conv1(inputs))
        s2 = F.max_pool2d(c1,(2,2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3,2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = F.relu(self.fc3(f6))
        return output

network = net()
network.to('cuda')
print(net)
        
## Train a network with MNIST

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

# Define loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(),lr=0.001,momentum = 0.9)

print("Loss function setted")


# Train the network
print('start training')
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

print("Finished training")

# Save the network
PATH = './cifar_net.pth'
torch.save(network.state_dict(), PATH)
        
