from enum import Enum
import torch as th
from torch import nn, optim
from torchvision import datasets, transforms
from abc import ABC, abstractmethod

class DatasetOrigin(Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    FASHION_MNIST = "FashionMNIST"
    CIFAR100 = "CIFAR100"
    IMAGENET = "ImageNet"

class Dataset(ABC):
    def __init__(self, origin: DatasetOrigin, transform=None):
        self.origin = origin
        self.transform = transform or self._default_transform()
        self.trainset = None
        self.testset = None
        self.num_classes = None
        self._load_data()

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def _default_transform(self):
        pass

    def get_train_loader(self, batch_size=64, shuffle=True):
        return th.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(self, batch_size=64, shuffle=False):
        return th.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=shuffle)

class MNISTDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__(DatasetOrigin.MNIST, transform)

    def _load_data(self):
        self.trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=self.transform)
        self.testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=self.transform)
        self.num_classes = 10

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

class CIFAR10Dataset(Dataset):
    def __init__(self, transform=None):
        super().__init__(DatasetOrigin.CIFAR10, transform)

    def _load_data(self):
        self.trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=self.transform)
        self.testset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=self.transform)
        self.num_classes = 10

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

class FashionMNISTDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__(DatasetOrigin.FASHION_MNIST, transform)

    def _load_data(self):
        self.trainset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=True, train=True, transform=self.transform)
        self.testset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=True, train=False, transform=self.transform)
        self.num_classes = 10

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

class CIFAR100Dataset(Dataset):
    def __init__(self, transform=None):
        super().__init__(DatasetOrigin.CIFAR100, transform)

    def _load_data(self):
        self.trainset = datasets.CIFAR100('~/.pytorch/CIFAR100_data/', download=True, train=True, transform=self.transform)
        self.testset = datasets.CIFAR100('~/.pytorch/CIFAR100_data/', download=True, train=False, transform=self.transform)
        self.num_classes = 100

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        super().__init__(DatasetOrigin.IMAGENET, transform)

    def _load_data(self):
        self.trainset = datasets.ImageNet(self.root, split='train', download=True, transform=self.transform)
        self.testset = datasets.ImageNet(self.root, split='val', download=True, transform=self.transform)
        self.num_classes = 1000

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with th.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return correct / total

def train_on_dataset(net, dataset, trainer=None, epochs=5, batch_size=64, learning_rate=0.001):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    net = net.to(device)

    if trainer is None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        trainer = Trainer(net, criterion, optimizer, device)

    train_loader = dataset.get_train_loader(batch_size=batch_size)
    test_loader = dataset.get_test_loader(batch_size=batch_size)

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        accuracy = trainer.evaluate(test_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("Training completed")