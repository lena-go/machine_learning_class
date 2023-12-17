import os

import torch
import torchvision
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    ConcatDataset,
)

import hyper_parameters


class MinusDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return 1500

    def __getitem__(self, item):
        img_path = os.path.join(
            self.img_dir,
            f"minus-{str(item).rjust(4, '0')}.png"
        )
        image = read_image(img_path)
        label = 10
        if self.transform:
            image = self.transform(image)
        return image, label


def get_loaders() -> (DataLoader, DataLoader):
    digits_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    minus_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomInvert(1.0),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_digits = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=digits_transforms
    )
    test_digits = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=digits_transforms
    )
    dataset_minus = MinusDataset(
        os.path.join('data', 'minus'),
        minus_transforms
    )
    train_test_ratio = round(len(test_digits) / len(train_digits), 1)
    train_minus, test_minus = random_split(
        dataset_minus,
        [1 - train_test_ratio, train_test_ratio]
    )
    train_dataset = ConcatDataset([train_digits, train_minus])
    test_dataset = ConcatDataset([test_digits, test_minus])

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
