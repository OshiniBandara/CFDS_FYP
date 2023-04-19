
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the Net class

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3) # Increase number of channels, reduce kernel size
        self.bn1 = nn.BatchNorm2d(12) # Batch normalization layer after each convolutional layer
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 3) # Increase number of channels, reduce kernel size
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(24, 48, 3) # Increase number of channels, reduce kernel size
        self.bn3 = nn.BatchNorm2d(48)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 2 * 2, 512) # Increase number of neurons in fully connected layer
        self.dropout1 = nn.Dropout(p=0.01)
        self.fc2 = nn.Linear(512, 100)
       

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 48 * 2 * 2)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


def train():
    # Define the transformation to be applied to the CIFAR-100 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Data augmentation: random crop with padding
        transforms.RandomHorizontalFlip(), # Data augmentation: random horizontal flip
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                             download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    # Define the device to be used for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    # Initialize the network and move it to the device
    net = Net().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

    # Train the network
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    # Evaluate the network on test images
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (100 * correct / total)
    print('Accuracy of the network on the 10000 test images: %.2f%%' % accuracy)


if __name__ == "__main__":
    train()