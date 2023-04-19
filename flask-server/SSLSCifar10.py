
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json

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
        self.dropout1 = nn.Dropout(p=0.001)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=0.001)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(84, 3*32*32)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = nn.functional.relu(self.fc2(x))
        self_supervised_output = self.fc4(x)
        self_supervised_output = self_supervised_output.view(-1, 3, 32, 32)
        x = self.fc3(x)
        return x, self_supervised_output

    
def ssls1():
    output = []
    # Define the transformation to be applied to the CIFAR-10 dataset
    transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4), # Data augmentation: random crop with padding
                transforms.RandomHorizontalFlip(), # Data augmentation: random horizontal flip
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define the device to be used for training
 
    # Initialize the network and move it to the device
    net = Net()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    self_supervised_criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    
    # Train the network
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs, labels

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, self_supervised_output = net(inputs)
            
            loss = criterion(outputs, labels)

            transformed_inputs = torch.rot90( inputs, 1 , [2,3] )

            self_supervised_loss = self_supervised_criterion(self_supervised_output, transformed_inputs)
            # Add the self-supervised signal loss to the total loss with a coefficient of 0.2
            total_loss = loss + 0.2 * self_supervised_loss
            total_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    # Evaluate the network on test images
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            #images, labels = images, labels
            outputs, _ = net(images)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy= float(correct)/float(total)*100
    print('Accuracy of the network on the 10000 test images: %.2f ' % accuracy)
    accuracy1=float(correct)/float(total)*100
    output.append(f'Accuracy of the network on the 10000 test images: %.2f%%' % accuracy1)
    json_data = json.dumps(str(output[0]))
    return json_data  
def main():
    ssls1()
if __name__ == "__main__":
    main()
