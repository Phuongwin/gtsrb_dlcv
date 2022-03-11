import argparse

# Machine Learning Libraries
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import csv

import torch.nn as nn
import torch.optim as optim

from preprocessing import *
from visualize import *
from models.basic_cnn import Net

def split_dataset(dataset: torchvision.datasets.folder.ImageFolder, train_size: float):
    if train_size > 1.0:
        raise Exception("Sizes do not equate to 1.0")
    
    train_set_int = int(len(dataset) * train_size)
    test_val_int = int(len(dataset) - train_set_int)
    val_set_int = int(test_val_int * 0.5)
    test_set_int = int(test_val_int * 0.5)

    train, val, test = torch.utils.data.random_split(dataset, [train_set_int, val_set_int, test_set_int])
    return train, val, test

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='GTSRB Classification - JHU DL for CV')

    # parser.add_argument('--batch-size', type=int, default=32, metavar='N',
    #                     help="input batch size for training (default: 32)")
    
    ### Loading the GTSRB dataset

    batch_size = 32
    train_set_allocation = 0.8

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.3403, 0.3121, 0.3214),
                                                        (0.2724, 0.2608, 0.2669))
                                  ])

    dataset = datasets.ImageFolder('./data/Final_Training/Images', transform = transform)

    view_data_distribution(dataset, "data_distribution")     # From visualize.py

    balanced_dataset = handle_imbalances(dataset)    # From preprocessing.py

    view_data_distribution(balanced_dataset, "balanced_distribution")

    # train_set, val_set, test_set = split_dataset(dataset, train_set_allocation)





    ### Load data into DataLoaders

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,
    #                                            num_workers = 2,
    #                                            drop_last = True,
    #                                            shuffle = True
    #                                           )

    # val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size,
    #                                          num_workers = 2,
    #                                          drop_last = True,
    #                                          shuffle = True
    #                                         )

    # test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size,
    #                                          num_workers = 2,
    #                                          drop_last = True,
    #                                          shuffle = True
    #                                         )
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])


    # net = Net()

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(),
    #                       lr=0.001,
    #                       momentum=0.9)
    
    # loss_plot = []

    # for epoch in range(2):
    #     print("training round ", epoch)

    #     running_loss = 0.0
    #     correct = 0
    #     total = 0

    #     for i, data in enumerate(train_loader, 0):
    #         inputs, labels = data

    #         optimizer.zero_grad()

    #         outputs = net(inputs)

    #         _, predicted = torch.max(outputs.data, 1)
    #         loss = criterion(outputs, labels)
    #         loss_plot.append(loss)

    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #         if i % 200 == 199:
    #             print('[%d, %5d] loss: %.3f accuracy: %.3f' %
    #               (epoch + 1, i + 1, running_loss / 200,100 * correct / total ))

    #             running_loss = 0.0


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
