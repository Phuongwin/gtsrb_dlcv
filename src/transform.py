import torch
from torchvision import datasets, transforms

class MapDataset(torch.utils.data.Dataset):
    ### Adapted from https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580/6
    
    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        if self.map:     
            x = self.map(self.dataset[index][0]) 
        else:     
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]   # label      
        return x, y

    def __len__(self):
        return len(self.dataset)

standard_transform = transforms.Compose([
                        transforms.Resize([32, 32]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),                             
                        ])

data_jitter_transform = transforms.Compose([
                        transforms.Resize([32, 32]),
                        transforms.ColorJitter(brightness=(0, 5)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),                             
                        ])

data_jitter_saturation = transforms.Compose([
                        transforms.Resize([32, 32]),
                        transforms.ColorJitter(saturation=(0,5)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),  
                        ])

flip_transform = transforms.Compose([transforms.Resize([32, 32]),
                        transforms.RandomRotation(10),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                        ])