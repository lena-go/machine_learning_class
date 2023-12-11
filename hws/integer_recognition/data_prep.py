import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import hyper_parameters


def get_loaders() -> (DataLoader, DataLoader):
    transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyper_parameters.BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hyper_parameters.BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader
