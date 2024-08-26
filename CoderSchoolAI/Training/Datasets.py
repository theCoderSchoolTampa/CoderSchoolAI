import logging
from enum import Enum
import torch as th
from torch import nn, optim
from torchvision import datasets, transforms
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetOrigin(Enum):
    """Natively supported dataset origins."""
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    FASHION_MNIST = "FashionMNIST"
    CIFAR100 = "CIFAR100"
    IMAGENET = "ImageNet"
    CUSTOM = ""

class Dataset(ABC):
    """
    Abstract base class for datasets.
    
    Attributes:
        origin (DatasetOrigin): The origin of the dataset.
        transform (callable): The transformation to be applied to the data.
        trainset (torch.utils.data.Dataset): The training dataset.
        testset (torch.utils.data.Dataset): The test dataset.
        num_classes (int): The number of classes in the dataset.
    """

    def __init__(self, origin: DatasetOrigin, transform: Optional[callable] = None):
        """
        Initialize the Dataset.

        Args:
            origin (DatasetOrigin): The origin of the dataset.
            transform (callable, optional): The transformation to be applied to the data.
        """
        self.origin = origin
        self.transform = transform or self._default_transform()
        self.trainset = None
        self.testset = None
        self.num_classes = None
        self._load_data()

    @abstractmethod
    def _load_data(self):
        """Abstract method to load the dataset."""
        pass

    @abstractmethod
    def _default_transform(self):
        """Abstract method to define the default transform."""
        pass

    def get_train_loader(self, batch_size: int = 64, shuffle: bool = True):
        """
        Get the data loader for the training set.

        Args:
            batch_size (int): The batch size for the data loader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            torch.utils.data.DataLoader: The data loader for the training set.
        """
        return th.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(self, batch_size: int = 64, shuffle: bool = False):
        """
        Get the data loader for the test set.

        Args:
            batch_size (int): The batch size for the data loader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            torch.utils.data.DataLoader: The data loader for the test set.
        """
        return th.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=shuffle)

class MNISTDataset(Dataset):
    """Dataset class for MNIST."""

    def __init__(self, transform: Optional[callable] = None):
        """
        Initialize the MNIST dataset.

        Args:
            transform (callable, optional): The transformation to be applied to the data.
        """
        super().__init__(DatasetOrigin.MNIST, transform)

    def _load_data(self):
        """Load the MNIST dataset."""
        self.trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=self.transform)
        self.testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=self.transform)
        self.num_classes = 10

    def _default_transform(self):
        """Define the default transform for MNIST."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

class CIFAR10Dataset(Dataset):
    """Dataset class for CIFAR-10."""

    def __init__(self, transform: Optional[callable] = None):
        """
        Initialize the CIFAR-10 dataset.

        Args:
            transform (callable, optional): The transformation to be applied to the data.
        """
        super().__init__(DatasetOrigin.CIFAR10, transform)

    def _load_data(self):
        """Load the CIFAR-10 dataset."""
        self.trainset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=True, transform=self.transform)
        self.testset = datasets.CIFAR10('~/.pytorch/CIFAR10_data/', download=True, train=False, transform=self.transform)
        self.num_classes = 10

    def _default_transform(self):
        """Define the default transform for CIFAR-10."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

class FashionMNISTDataset(Dataset):
    """Dataset class for Fashion-MNIST."""

    def __init__(self, transform: Optional[callable] = None):
        """
        Initialize the Fashion-MNIST dataset.

        Args:
            transform (callable, optional): The transformation to be applied to the data.
        """
        super().__init__(DatasetOrigin.FASHION_MNIST, transform)

    def _load_data(self):
        """Load the Fashion-MNIST dataset."""
        self.trainset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=True, train=True, transform=self.transform)
        self.testset = datasets.FashionMNIST('~/.pytorch/FashionMNIST_data/', download=True, train=False, transform=self.transform)
        self.num_classes = 10

    def _default_transform(self):
        """Define the default transform for Fashion-MNIST."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

class CIFAR100Dataset(Dataset):
    """Dataset class for CIFAR-100."""

    def __init__(self, transform: Optional[callable] = None):
        """
        Initialize the CIFAR-100 dataset.

        Args:
            transform (callable, optional): The transformation to be applied to the data.
        """
        super().__init__(DatasetOrigin.CIFAR100, transform)

    def _load_data(self):
        """Load the CIFAR-100 dataset."""
        self.trainset = datasets.CIFAR100('~/.pytorch/CIFAR100_data/', download=True, train=True, transform=self.transform)
        self.testset = datasets.CIFAR100('~/.pytorch/CIFAR100_data/', download=True, train=False, transform=self.transform)
        self.num_classes = 100

    def _default_transform(self):
        """Define the default transform for CIFAR-100."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

class ImageNetDataset(Dataset):
    """Dataset class for ImageNet."""

    def __init__(self, root: str, transform: Optional[callable] = None):
        """
        Initialize the ImageNet dataset.

        Args:
            root (str): The root directory of the ImageNet dataset.
            transform (callable, optional): The transformation to be applied to the data.
        """
        self.root = root
        super().__init__(DatasetOrigin.IMAGENET, transform)

    def _load_data(self):
        """Load the ImageNet dataset."""
        self.trainset = datasets.ImageNet(self.root, split='train', download=True, transform=self.transform)
        self.testset = datasets.ImageNet(self.root, split='val', download=True, transform=self.transform)
        self.num_classes = 1000

    def _default_transform(self):
        """Define the default transform for ImageNet."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class Trainer:
    """
    A class to handle the training and evaluation of a model.

    Attributes:
        model (nn.Module): The neural network model to be trained.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimization algorithm.
        device (torch.device): The device to run the computations on.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, device: th.device):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The neural network model to be trained.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimization algorithm.
            device (torch.device): The device to run the computations on.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, train_loader: th.utils.data.DataLoader) -> float:
        """
        Train the model for one epoch.

        Args:
            train_loader (DataLoader): The data loader for the training set.

        Returns:
            float: The average loss for this epoch.
        """
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

    def evaluate(self, test_loader: th.utils.data.DataLoader) -> float:
        """
        Evaluate the model on the test set.

        Args:
            test_loader (DataLoader): The data loader for the test set.

        Returns:
            float: The accuracy of the model on the test set.
        """
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

def train_on_dataset(net: nn.Module, dataset: Dataset, trainer: Optional[Trainer] = None, 
                     epochs: int = 5, batch_size: int = 64, learning_rate: float = 0.001):
    """
    Train a neural network on a given dataset.

    Args:
        net (nn.Module): The neural network to train.
        dataset (Dataset): The dataset to train on.
        trainer (Trainer, optional): A pre-configured Trainer object.
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size for training.
        learning_rate (float): The learning rate for the optimizer.
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    net = net.to(device)

    if trainer is None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        trainer = Trainer(net, criterion, optimizer, device)

    train_loader = dataset.get_train_loader(batch_size=batch_size)
    test_loader = dataset.get_test_loader(batch_size=batch_size)

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        accuracy = trainer.evaluate(test_loader)
        train_losses.append(train_loss)
        test_accuracies.append(accuracy)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")

    logger.info("Training completed")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    logger.info("Training progress plot saved as 'training_progress.png'")