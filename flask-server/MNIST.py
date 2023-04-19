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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 12)
        self.fc1 = nn.Linear(864, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train3():
    output = []
    transform1 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform1)
    trainloader= torch.utils.data.DataLoader(trainset, batch_size=100,
                                            shuffle=True, num_workers=1)
    train_size1 = int(0.5 * len(trainset))
    train_size2 = len(trainset)-int(0.5 * len(trainset))
    trainset1, trainset2 = torch.utils.data.random_split(trainset, [train_size1, train_size2])

    trainloader1= torch.utils.data.DataLoader(trainset1, batch_size=100,
                                            shuffle=True, num_workers=1)
    trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=100,
                                            shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                            shuffle=False, num_workers=1)
    classes = ('1','2','3','4','5','6','7','8','9','0')

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader1, 0):
            # get the inputs
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy1= float(correct)/float(total)*100
    #print('Accuracy of the network on the 10000 test images: %.2f ' % accuracy1)
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct_indicator = [0 for i in range(len(testset))]
    i = 0
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    correct_indicator[i] = 1
                total_pred[classes[label]] += 1
                i += 1

    # 1 - 1, 2 - 1, 3 - 1, 4-0, 5-0, 6-0 - 60
    # 2, 3, 4, 5 - 80
    #print(correct_indicator[0:20])

    accdict1 = {}
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracyTest1 = 100 * float(correct_count) / total_pred[classname]
        accdict1[classname] = accuracyTest1
        #print(f'Accuracy for class: {classname:5s} is {accuracyTest1:.1f} %')

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader2, 0):
            # get the inputs
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


    correct = 0
    total = 0
    print(len(testset))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy2= float(correct)/float(total)*100
    #print(total)
    #print('Accuracy of the network on the 10000 test images: %.2f ' % accuracy2)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct_indicator = [0 for i in range(len(testset))]
    i = 0
    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    correct_indicator[i] = 2
                total_pred[classes[label]] += 1
                i += 1

    # 1 - 1, 2 - 1, 3 - 1, 4-0, 5-0, 6-0 - 60
    # 2, 3, 4, 5 - 80
    #print(correct_indicator[0:20])

    accdict2 = {}
    for classname, correct_count in correct_pred.items():

        accuracyTest2 = 100 * float(correct_count) / total_pred[classname]
        accdict2[classname] = accuracyTest2
        #print(f'Accuracy for class: {classname:5s} is {accuracyTest2:.1f} %')

    #forgetfullness = (accuracy1-accuracy2)
    #if forgetfullness < 0:
        #print('forgetting percentage of the network is: 0')
    #else:
        #print('forgetting percentage of the network is: %f %%' % forgetfullness)     
    

    result = {}
    for key in accdict1:
        if key in accdict2:
            percentage_change = 0 # initialize with default value
            if (accdict1[key] - accdict2[key]) > 0:
                percentage_change = round((accdict1[key] - accdict2[key]), 2)
            if percentage_change > 0:
                result[key] = str(percentage_change) + "%"

    result_list = [f'{k}: {v}' for k, v in result.items()]

 
    output.extend(result_list)

    return output

def main():
    train3()
if __name__ == "__main__":
    main()