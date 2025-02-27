# -*- coding: utf-8 -*-
"""Copy of BEST HACK'25 DATA SCIENCE.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BOJnnRERwuZquoJmvD-LSiu-ijyjx2wK

Привет!

В этом ноутбуке написан полный процесс обучения модели классификации картинок. Кроме того, здесь спрятаны несколько ошибок: как синтаксические, так и логические.

Ваша задача
1. Добиться, чтобы исходный ноутбук корректно запускался, без ошибок при исполнении колонок.
2. Найти как можно больше логических ошибок. А именно тех ошибок, исправление которых может привести к увеличению тестовой метрики.
Тестовая метрика в данной задаче - accuracy

Условия при исправлении ошибок:
- нельзя переписывать процесс формирования данных с нуля, можно исправлять и дополнять существующий код
- нельзя переписывать архитектуру модели с нуля, можно исправлять и дополнять существующий код

# Загрузка библиотек
"""

# Commented out IPython magic to ensure Python compatibility.
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['font.size'] = 12

from tqdm.auto import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, Subset

"""# Загрузка функций"""

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

"""# Процесс формирования данных"""

!gdown https://drive.google.com/uc?id=1JuaZXdcY_pADHRzSD3QjHDEpTgCYjL62

data = torch.load('data.pth')
train = data['train']
test = data['test']

len(train['images']), len(test['images'])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class PreprocessedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.images = dataset['images']
        self.labels = dataset['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = (self.images[idx],
                        self.labels[idx])

        # Apply transformation if defined
        if self.transform:
            image = self.transform(image)

        return image, label

train = PreprocessedDataset(
    train,
    transform=train_transform
)

test = PreprocessedDataset(
    test,
    transform=test_transform
)

@dataclass
class TrainConfig:
    num_workers: int = 4
    batch_size: int = 32
    valid_size: float = 0.2

    n_labels: int = 6

    n_epochs: int = 10
    lr: float = 1e-3

    dropout: float = 0.95

config = TrainConfig()

num_train = len(train)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(config.valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=config.batch_size,
    sampler=train_sampler,
    num_workers=config.num_workers
)

valid_loader = torch.utils.data.DataLoader(
    train,
    batch_size=config.batch_size,
    sampler=valid_sampler,
    num_workers=config.num_workers
)

test_loader = torch.utils.data.DataLoader(
    train,
    batch_size=config.batch_size,
    num_workers=config.num_workers
)

"""# Архитектура модели"""

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, config.n_labels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 4*4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

"""# Обучение"""

for epoch in range(1, config.n_epochs + 1):
    train_loss = 0
    valid_loss = 0

    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    print(
        f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}'
    )

"""# Валидация на тестовой выборке"""

test_loss = 0
class_correct = list(0 for i in range(config.n_labels))
class_total = list(0 for i in range(config.n_labels))

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = pred.eq(target)
        # correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(
        #     correct_tensor.numpy() if device == 'cpu' else np.squeeze(correct_tensor.cpu().numpy())
        # )

        for i in range(len(target)):
            label = target[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

test_loss = test_loss / len(test_loader.dataset)
print(f'Test loss: {test_loss:.6f}')

print(
    'Test accuracy: %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total))
)