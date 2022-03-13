import argparse
import time
import matplotlib.pyplot as plt

# Machine Learning Libraries
import torch
import torchvision
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim

# Other .py files to be used
from preprocessing import *
from visualize import *
from models.basic_cnn import Net

def split_dataset(dataset: torchvision.datasets.folder.ImageFolder, train_percentage: float):
    if train_percentage > 1.0:
        raise Exception("Sizes do not equate to 1.0")
    
    dataset_size = dataset.__len__()

    train_count = int(dataset_size * train_percentage)
    test_count = dataset_size - train_count


    train_set, test_set = torch.utils.data.random_split(dataset, [train_count, test_count])
    print(train_set.__len__(), test_set.__len__())
    print(f"Confirming total: {train_set.__len__() + test_set.__len__()}")

    return train_set, test_set

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='GTSRB Classification - JHU DL for CV')

    # parser.add_argument('--batch-size', type=int, default=32, metavar='N',
    #                     help="input batch size for training (default: 32)")
    
    ### Define Hyperparameters to be used within command line arguments
    batch_size = 32
    train_set_allocation = 0.9

    PATH = './saved_models/custom_network.pth'

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                  ])

    ### Loading the GTSRB dataset
    dataset = datasets.ImageFolder('./data/Final_Training/Images', transform = transform)

    train_set, test_set = split_dataset(dataset, train_set_allocation)

    view_data_distribution(dataset, "data_distribution")     # From visualize.py

    sampler = make_random_weight_sampler(dataset, train_set)    # From preprocessing.py


    ### Load data into DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size = batch_size,
                                               sampler = sampler,
                                               num_workers = 2,
                                               drop_last = True,
                                              )

    test_loader = torch.utils.data.DataLoader(test_set, 
                                              batch_size = batch_size,
                                              num_workers = 2,
                                              drop_last = True,
                                              shuffle = True
                                             )

    ### Instantiate Model and prepare for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Net()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.001,
                          momentum=0.9)
    
    loss_plot = []
    t1 = time.perf_counter()

    for epoch in range(4):
        print("training round ", epoch)

        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            input, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss_plot.append(loss)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200,100 * correct / total ))

                running_loss = 0.0
    t2 = time.perf_counter()

    print(f"Finished Training in {int(t2 - t1)} seconds")

    torch.save(model.state_dict(), PATH)

    ### Inference/Testing
    print("begin inferencing")
    model.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0

    n_classes = 43
    confusion_matrix = torch.zeros(n_classes, n_classes)

    with torch.no_grad():
        for index, data in enumerate(test_loader):
            images, labels = data
            
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    save_confusion_matrix(confusion_matrix)


    # print('Finished Training')
    # plt.plot(range(len(loss_plot)),loss_plot, 'r+')
    # plt.title("Loss")
    # plt.show()

    # print(net)
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # https://github.com/gautam-sharma1/GTSRB-torch/blob/master/main.py

    # figure = plt.figure(figsize=(8, 8))
    # cols, rows = 5, 5

    # for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_loader), size= (1,)).item()
    #     img, label = dataset[sample_idx]
    #     figure.add_subplot(rows, cols, i)
    #     plt.title(label)
    #     plt.axis('off')
    #     plt.imshow(img, cmap="gray")
    # plt.show()
