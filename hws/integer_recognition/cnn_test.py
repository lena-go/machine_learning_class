import torch.nn as nn

from helpers import plot
from data_prep import get_loaders


# get some random training images
train_loader = get_loaders()[0]
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
plot(images)

conv1 = nn.Conv2d(3, 6, 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)
print(images.shape)
x = conv1(images)
print(x.shape)
x = pool(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = pool(x)
print(x.shape)
