# This is a code testing the network
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Loading test data set
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
     ])

batch_size = 1
test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=0)



# Declear a network
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
PATH = './cifar_net.pth'
network.load_state_dict(torch.load(PATH,weights_only = True))
network.to('cuda')

# Testing the data set
accuracy = 0
for data in testloader:
    image, labels = data[0].to('cuda'),data[1].to('cuda')
    output = network(image)
    if torch.argmax(output) == labels:
        accuracy +=1
accuracy = 100*accuracy / len(testloader)
print(f"The accuracy of the network is: {accuracy}%")
