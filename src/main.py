# Generic Python Libraries
import yaml
import time

# Machine Learning Libraries
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Other .py files to be used
from preprocessing import *
from visualize import *
from models.basic_cnn import Net
from transform import *
if __name__ == "__main__":
    ### Read Configuration and Hyperparameters from config.yaml
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    print(config)

    # Configurations
    train = config['training']
    inference = config['inference']
    handle_imbalances = config['imbalances']
    dataset_root = config['dataset_path']
    save_path = config['save_path']
    PATH = config['weight_path']

    # Hyperparameters
    train_set_allocation = config['train_allocation']
    batch_size = config['batch_size']
    n_epoch = config['epoch']
    learning_rate = config['learning_rate']
    momentum = config['momentum']

    # Hardware for Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    '''
    Loading the GTSRB dataset
    '''
    dataset = datasets.ImageFolder(dataset_root)   
    train_set, valid_set, test_set = split_dataset(dataset, train_set_allocation) # From preprocessing.py

    view_data_distribution(dataset, "data_distribution")     # From visualize.py

    '''
    Data Preprocessing
    '''
    if (handle_imbalances): ### Part 2: Data Augmentation and Weighted Random Sampler
        # Augment training data and add into train_loader
        train_aug_1 = MapDataset(train_set, standard_transform)
        train_aug_2 = MapDataset(train_set, data_jitter_transform)
        train_aug_3 = MapDataset(train_set, data_jitter_saturation)
        train_aug_4 = MapDataset(train_set, flip_transform)

        print(f"Augmentations performed: {standard_transform} {data_jitter_transform} {data_jitter_saturation} {flip_transform}")
        concat_train_set = torch.utils.data.ConcatDataset([train_aug_1, train_aug_2, train_aug_3, train_aug_4])
        
        sampler = make_weighted_random_sampler(dataset, train_set)    # From preprocessing.py

        train_loader = torch.utils.data.DataLoader(concat_train_set, 
                                                batch_size = batch_size,
                                                sampler = sampler,
                                                num_workers = 2,
                                                drop_last = True,
                                                )
    
    else: ### Part 1: Standard pre-processing
        train_aug = MapDataset(train_set, standard_transform)
        print(f'Augmentations performed: {standard_transform}')

        train_loader = torch.utils.data.DataLoader(train_aug, 
                                                batch_size = batch_size,
                                                num_workers = 2,
                                                drop_last = True,
                                                shuffle = True
                                                )

    ### Standard augmentation and loading for Validation and Test sets
    valid_aug = MapDataset(valid_set, standard_transform)
    test_aug = MapDataset(test_set, standard_transform)

    valid_loader = torch.utils.data.DataLoader(valid_aug, 
                                               batch_size = batch_size,
                                               num_workers = 2,
                                               drop_last = True,
                                               shuffle = True
                                             )

    test_loader = torch.utils.data.DataLoader(test_aug, 
                                              batch_size = batch_size,
                                              num_workers = 2,
                                              drop_last = True,
                                              shuffle = True
                                             )

    '''
    Instantiate Model
    '''
    model = Net()   # imported from basic_cnn.py
    print(model)

    '''
    Training Procedures
    '''
    if (train): # Configured through config.yaml
        print("Defining Loss function and Optimizer")
        ### Define Model's loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,     # Defined in config.yaml
                              momentum=momentum)    # Defined in config.yaml
        
        ### Define Lists for visualizations
        train_loss_plot = []
        train_acc_plot = []
        valid_loss_plot = []
        valid_acc_plot = []
        print("Begin Training")
        t1 = time.perf_counter()
        for epoch in range(n_epoch):
            print("Training round ", epoch + 1)

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

        torch.save(model.state_dict(), save_path)

        epoch_count = range(1, n_epoch + 1)

        if (handle_imbalances):
            plot_train_validation(epoch_count, train_acc_plot, valid_acc_plot, "Enhanced", 'Accuracy')
            plot_train_validation(epoch_count, train_acc_plot, valid_acc_plot, "Enhanced", 'Loss')
        else:
            plot_train_validation(epoch_count, train_acc_plot, valid_acc_plot, "Basic", 'Accuracy')
            plot_train_validation(epoch_count, train_acc_plot, valid_acc_plot, "Basic", 'Loss')

    '''
    Inferencing
    '''
    if (inference and PATH):
        print("Begin Inferencing")
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

        if (handle_imbalances):
            save_confusion_matrix(confusion_matrix, 'Enhanced')
        else: 
            save_confusion_matrix(confusion_matrix, 'Basic')

    print("End Script")

