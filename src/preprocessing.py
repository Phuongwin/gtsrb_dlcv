import torchvision
import torch
from collections import Counter
import numpy as np

def split_dataset(dataset: torchvision.datasets.folder.ImageFolder, train_percentage: float = 0.8):
    if train_percentage > 1.0:
        raise Exception("Inputted Train Percentage is too large")
    
    dataset_size = dataset.__len__()

    train_count = int(dataset_size * train_percentage)
    valid_test_count = dataset_size - train_count
    
    valid_count = int(valid_test_count * 0.5)
    test_count = valid_test_count - valid_count


    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [train_count, valid_count, test_count])
    
    print(f'Train Set: {train_set.__len__()}')
    print(f'Validation Set: {valid_set.__len__()}')
    print(f'Test Set: {test_set.__len__()}')
    print(f"Total: {train_set.__len__() + valid_set.__len__() + test_set.__len__()}")

    return train_set, valid_set, test_set

def make_weighted_random_sampler(dataset: torchvision.datasets.folder.ImageFolder, train_set: torch.utils.data.dataset.Subset):
    #https://cs230.stanford.edu/projects_fall_2020/reports/55824835.pdf 
    # Potential Approach is to use WeightedRandomSample
    #https://towardsdatascience.com/getting-started-with-albumentation-winning-deep-learning-image-augmentation-technique-in-pytorch-47aaba0ee3f8
    
    y_train_indices = train_set.indices
    y_train = [dataset.targets[i] for i in y_train_indices]

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    weights = 1. / class_sample_count
    samples_weight = np.array([weights[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights = samples_weight.type('torch.DoubleTensor'),
        num_samples = len(samples_weight)
    )

    return sampler