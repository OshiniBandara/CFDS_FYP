import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import confusion_matrix

# Define the Net class
SEED = 1000
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def normalCifar10():
    output =[]
    transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

   

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    """ accuracies = []
    f_measures = [] """

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    # mse_loss = 0
    # num_classes=10
    # all_labels = []
    # all_predictions = [] 

    with torch.no_grad():
        for data in testloader:
            images, labels = data
                # calculate outputs by running images through the network
            outputs = net(images)
                # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
                #all_labels.extend(labels.numpy())
                #all_predictions.extend(predicted.numpy()) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                #target = nn.functional.one_hot(labels, num_classes=num_classes)
                #mse_loss += nn.MSELoss()(outputs, target.float())
            accuracy = float(correct)/float(total)*100
            #accuracies.append(accuracy)
        print(f'Accuracy of the network on the 10000 test images: %.2f ' % accuracy)
        accuracy1=float(correct)/float(total)*100
        output.append(f'Accuracy of the network on the 10000 test images: %.2f%%' % accuracy1)
        json_data = json.dumps(str(output[0]))


       
        
    #return json_data
        

    return json_data  
def main():
    normalCifar10()

if __name__ == "__main__":
    main()
