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

def split_dataset(dataset: torchvision.datasets.folder.ImageFolder, train_percentage: float = 0.8):
    if train_percentage > 1.0:
        raise Exception("Inputted Train Percentage is too large")
    
    dataset_size = dataset.__len__()

    train_count = int(dataset_size * train_percentage)
    valid_test_count = dataset_size - train_count

    valid_count = int(valid_test_count * 0.5)
    test_count = int(valid_test_count * 0.5)


    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_count, valid_count, test_count])
    
    print(train_set.__len__(), valid_set.__len__(), test_set.__len__())
    print(f"Confirming total: {train_set.__len__() + valid_set.__len__() + test_set.__len__()}")

    return train_set, valid_set, test_set

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='GTSRB Classification - JHU DL for CV')

    # parser.add_argument('--batch-size', type=int, default=32, metavar='N',
    #                     help="input batch size for training (default: 32)")
    
    ### Define Hyperparameters to be used within command line arguments
    batch_size = 32
    train_set_allocation = 0.8
    n_epoch = 2

    PATH = './saved_models/custom_network.pth'

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                  ])

    ### Loading the GTSRB dataset
    dataset = datasets.ImageFolder('./data/Final_Training/Images', transform = transform)

    train_set, valid_set, test_set = split_dataset(dataset, train_set_allocation)

    view_data_distribution(dataset, "data_distribution")     # From visualize.py

    sampler = make_random_weight_sampler(dataset, train_set)    # From preprocessing.py


    ### Load data into DataLoaders
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size = batch_size,
                                               sampler = sampler,
                                               num_workers = 2,
                                               drop_last = True,
                                              )

    valid_loader = torch.utils.data.DataLoader(valid_set, 
                                               batch_size = batch_size,
                                               num_workers = 2,
                                               drop_last = True,
                                               shuffle = True
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
    
    ### Define Lists for visualizations
    train_loss_plot = []
    train_acc_plot = []
    valid_loss_plot = []
    valid_acc_plot = []

    t1 = time.perf_counter()
    for epoch in range(n_epoch):
        print("Training round ", epoch)

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            input, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        model.eval()
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

            valid_loss += loss.item()

        print(f'Epoch {epoch + 1} \t Training Loss: {(train_loss / len(train_loader)):.4f} \
                               Training Acc: {(train_correct / train_total):.4f} \
                               Validation Loss: {(valid_loss / len(valid_loader)):.4f} \
                               Validation Acc: {(valid_correct / valid_total):.4f}')

        train_loss_plot.append(round(train_loss / len(train_loader), 4))
        train_acc_plot.append(round(train_correct / train_total, 4))
        valid_loss_plot.append(round(valid_loss / len(valid_loader), 4))
        valid_acc_plot.append(round(valid_correct / valid_total, 4))

    t2 = time.perf_counter()

    print(f"Finished Training in {int(t2 - t1)} seconds")

    torch.save(model.state_dict(), PATH)

    epoch_count = range(1, n_epoch + 1)
    plot_train_validation(epoch_count, train_acc_plot, valid_acc_plot, "Accuracy", 'train_valid_acc_plot')
    plot_train_validation(epoch_count, train_acc_plot, valid_acc_plot, "Loss", 'train_valid_loss_plot')

    ### Inference/Testing
    print("begin inferencing")
    model.load_state_dict(torch.load(PATH))

    test_correct = 0
    test_total = 0

    n_classes = 43
    confusion_matrix = torch.zeros(n_classes, n_classes)

    with torch.no_grad():
        for index, data in enumerate(test_loader):
            images, labels = data
            
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    print(f'Accuracy of the network on the {test_total} test images: {100 * test_correct // test_total} %')

    save_confusion_matrix(confusion_matrix)

