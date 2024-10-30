from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch

class TinyDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example['image']  
        label = example['label']  
        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            return None
        else:
            return image, label

def load_tiny(batch_size=64, shuffle=False, is_train=False):
    imgSize = 224
    toTensorTransform = transforms.Compose([
        transforms.Resize(imgSize),
        transforms.ToTensor(),
    ])
    tiny_imagenet_val = load_dataset('Maysee/tiny-imagenet', split='valid')
    val_dataset = TinyDataset(tiny_imagenet_val, transform=toTensorTransform)
    val_dataset = [item for item in val_dataset if item is not None]
    np.random.seed(0)
    val_index = np.random.choice(len(val_dataset), int(0.2*len(val_dataset)), replace=False)
    custom_val_dataset = [val_dataset[i] for i in val_index]
    val_loader = DataLoader(custom_val_dataset, batch_size=batch_size, shuffle=shuffle) 

    test_index = np.setdiff1d(np.arange(len(val_dataset)), val_index)
    custom_test_dataset = [val_dataset[i] for i in test_index]
    test_loader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=shuffle) 

    if is_train == True:
        tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
        custom_train_dataset = TinyDataset(tiny_imagenet_train, transform=toTensorTransform)
        custom_train_dataset = [item for item in custom_train_dataset if item is not None]
        train_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)    
        return train_loader, test_loader
    else:
        return val_loader

def GetCIFAR100Training(imgSize = 32, batchSize=64):
    toTensorTransform = transforms.Compose([
        transforms.Resize(imgSize),
        transforms.ToTensor(),
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='dataset/', train=True, download=True, transform=toTensorTransform), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader

def GetCIFAR100Validation(imgSize = 32, batchSize=64, ratio=1):
    transformTest = transforms.Compose([
        transforms.Resize(imgSize),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.CIFAR100(root= 'dataset/', train=False, download=True, transform=transformTest)
    np.random.seed(0)
    test_index = np.random.choice(len(val_dataset),int(ratio*len(val_dataset)),replace=False)
    val_dataset = [val_dataset[i] for i in test_index]
    valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

def GetCIFAR10Training(imgSize = 32, batchSize=64):
    toTensorTransform = transforms.Compose([
        transforms.Resize(imgSize),
        transforms.ToTensor(),
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='dataset/', train=True, download=True, transform=toTensorTransform), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader

def GetCIFAR10Validation(imgSize = 32, batchSize=64, ratio=1):
    transformTest = transforms.Compose([
        transforms.Resize(imgSize),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.CIFAR10(root='dataset/', train=False, download=True, transform=transformTest)
    np.random.seed(0)
    test_index = np.random.choice(len(val_dataset),int(ratio*len(val_dataset)),replace=False)
    val_dataset = [val_dataset[i] for i in test_index]
    valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader