import os
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from PIL import Image

from helpers import timeit
import hyper_parameters
from data_prep import get_loaders, prepare_image


MODEL_PATH = 'cnn.pt'


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # refer to cnn_test.py
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, hyper_parameters.NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # -1 is batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@timeit
def train(
        model: ConvNet,
        train_loader: DataLoader,
        device: torch.device,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.SGD,
):
    n_total_steps = len(train_loader)
    epoch_loss = [0 for _ in range(hyper_parameters.NUM_EPOCHS)]
    for epoch in range(hyper_parameters.NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2000 == 0:
                print(f'Epoch [{epoch + 1}/{hyper_parameters.NUM_EPOCHS}], '
                      f'Step [{i + 1}/{n_total_steps}], '
                      f'Loss: {loss.item():.4f}')

            epoch_loss[epoch] += loss.item()

    print('Finished Training')
    plt.plot(epoch_loss)
    torch.save(model.state_dict(), MODEL_PATH)


@timeit
def evaluate(
        model: ConvNet,
        test_loader: DataLoader,
        device: torch.device,
        classes: tuple,
):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(hyper_parameters.NUM_CLASSES)]
        n_class_samples = [0 for _ in range(hyper_parameters.NUM_CLASSES)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(hyper_parameters.BATCH_SIZE):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        total_acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {total_acc}%')

        for i in range(hyper_parameters.NUM_CLASSES):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc}%')


def get_model(device: torch.device) -> ConvNet:
    if not os.path.exists(MODEL_PATH):
        make_model(do_eval=False, device=device)
    model = ConvNet().to(device)
    cnn_state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(cnn_state_dict)
    return model


def classify_number(images: [Image]) -> str:
    print('Trying to recognize...')
    print('Total symbols -', len(images))
    device = get_device()
    model = get_model(device)
    model.eval()

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '-')
    number_idx = []
    read_up_to = hyper_parameters.BATCH_SIZE
    num_passes = ceil(len(images) / hyper_parameters.BATCH_SIZE)

    with torch.no_grad():
        for i in range(num_passes):
            batch = torch.zeros(
                hyper_parameters.BATCH_SIZE, 3, 28, 28,
                dtype=torch.float32,
                device=device,
            )
            for j in range(hyper_parameters.BATCH_SIZE):
                img_i = i*hyper_parameters.BATCH_SIZE + j
                if img_i < len(images):
                    batch[j] += prepare_image(
                        images[img_i],
                        device=device
                    )
                else:
                    read_up_to = j
                    break

            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            number_idx += predicted.tolist()[:read_up_to]

    return idx_to_number(classes, number_idx)


def idx_to_number(classes: tuple, number_idx: [int]) -> str:
    number = []
    for val in number_idx:
        number.append(str(classes[val]))
    return ''.join(number)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_model(rewrite: bool = False, do_eval: bool = True, device: torch.device = None):
    print('Device is', device)

    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'minus')

    train_loader, test_loader = get_loaders()
    model = ConvNet().to(device)
    if os.path.exists(MODEL_PATH) and not rewrite:
        print('Using pretrained model...')
        cnn_state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(cnn_state_dict)
    else:
        print('Starting training...')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hyper_parameters.LEARNING_RATE
        )
        train(model, train_loader, device, criterion, optimizer)
        plt.show()
    if do_eval:
        evaluate(model, test_loader, device, classes)


def run():
    device = get_device()
    make_model(device=device)


if __name__ == '__main__':
    run()
