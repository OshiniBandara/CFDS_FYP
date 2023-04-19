 
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import json
from torch.utils.data import DataLoader

def train_model(dataset_name, model_file):
    output = []
    if dataset_name == 'MNIST':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        class_names = ('1','2','3','4','5','6','7','8','9','0')
    elif dataset_name == 'CIFAR10':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
    elif dataset_name == 'CIFAR100':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform) 
        class_names = ('beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                       'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'bottles', 'bowls', 'cans', 'cups',
                       'plates', 'apples', 'mushrooms', 'oranges', 'pears', 'sweet_peppers', 'clock', 'computer_keyboard',
                       'lamp', 'telephone', 'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle',
                       'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 'tiger', 'wolf',
                       'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain',
                       'sea', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum',
                       'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 'baby', 'boy', 'girl', 'man',
                       'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 'rabbit',
                       'shrew', 'squirrel', 'maple', 'oak', 'palm', 'pine', 'willow', 'bicycle', 'bus', 'motorcycle',
                       'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor')
        
    else:
        return {'message': 'Invalid dataset name.'}

    train_size = int(0.5 * len(train_dataset))
    train_dataset1, train_dataset2 = torch.utils.data.random_split(train_dataset, [train_size, train_size])
    
    train_loader1 = DataLoader(train_dataset1, batch_size=64, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=64, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



    model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with open('Report.txt', 'a') as f:
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(train_loader1, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    f.write('[%d, %5d] loss: %.3f\n' %
                        (epoch + 1, i + 1, running_loss / 100) )
                    running_loss = 0.0

    # Save the model to a file
    #torch.save(model.state_dict(), model_file)

    
    # Evaluate the model on the test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy1 = 100.0 * correct / total
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    correct_indicator = [0 for i in range(len(test_dataset))]
    i = 0
    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                    correct_indicator[i] = 1
                total_pred[class_names[label]] += 1
                i += 1

    # print accuracy for each class
    accdict1 = {}
    print('1')
    #for key,values in accdict1:
    for classname, correct_count in correct_pred.items():

        accuracyTest1 = 100 * float(correct_count) / total_pred[classname]
        accdict1[classname] = accuracyTest1
    with open('Report.txt', 'a') as f:
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(train_loader2, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    f.write('[%d, %5d] loss: %.3f\n' %
                      (epoch + 1, i + 1, running_loss / 100) )
                    running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy2 = 100.0 * correct / total
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    correct_indicator = [0 for i in range(len(test_dataset))]
    i = 0
    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                
                if label == prediction:
                    correct_pred[class_names[label]] += 1
                    correct_indicator[i] = 2
                total_pred[class_names[label]] += 1
                i += 1


    # print accuracy for each class
    accdict2 = {}
    print('2')
    for classname, correct_count in correct_pred.items():

        accuracyTest2 = 100 * float(correct_count) / total_pred[classname]
        accdict2[classname] = accuracyTest2

    result = {}
    print('3')
    with open('Report.txt', 'a') as f:
        for key in accdict1:
            if key in accdict2:
                if (accdict1[key] - accdict2[key]) > 0:
                    result[key] = round(accdict1[key] - accdict2[key],2)
  
        [f.write(f'{key}:{value:+.2f}%\n') for key, value in result.items()]
   
        [print('{key}:{value:+.2f}%\n') for key, value in result.items()]
        json_data = json.dumps(result)   
    #return json_data
        output.append(json_data)
    #output.append('\n Forgetting Detected! \n')
    return output

