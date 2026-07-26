import torch
import torch.nn as nn

class fashion_dataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        one_hot_label = torch.zeros(10)  # there are 10 classes in the Fashion MNIST dataset
        one_hot_label[label] = 1
        return image, one_hot_label

