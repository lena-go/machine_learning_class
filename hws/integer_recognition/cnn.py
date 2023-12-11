import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from helpers import timeit
import hyper_parameters
from data_prep import get_loaders


MODEL_PATH = 'cnn.pt'


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # refer to cnn_test.py
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
        classes: (int,),
):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for _ in range(10)]
        n_class_samples = [0 for _ in range(10)]
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
        print(f'Accuracy of the network: {total_acc} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')


def run(rewrite: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is', device)

    classes = tuple(range(0, 10))

    train_loader, test_loader = get_loaders()
    model = ConvNet().to(device)
    if os.path.exists(MODEL_PATH) and not rewrite:
        print('Using pretrained model...')
        cnn_state_dict = torch.load(MODEL_PATH)
        model.load_state_dict(cnn_state_dict)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hyper_parameters.LEARNING_RATE
        )
        train(model, train_loader, device, criterion, optimizer)
        plt.show()
    evaluate(model, test_loader, device, classes)


if __name__ == '__main__':
    run()
