# Dataset y DataLoader para rEAMember
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import decode_image
import os
import sys
from PIL import Image
import numpy as np

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/'))
sys.path.append(path)
from SPOTS.utils import SPOT10Loader

# Custom Dataset

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=transforms.ToTensor(), target_transform=torch.tensor):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
            if img.ndim == 2:
                img = img.unsqueeze(0)
        if self.target_transform:
            target = self.target_transform(target)
        assert img.ndim == 3 , f"Shape incorrecto: {img.shape}"
        return img, target
        

class ImageDatasetWrapper:
    """
    Carga y expone train y test como atributos .train y .test
    """
    def __init__(self, dataset_name="FashionMNIST", data_path="./data", transform=None, custom_class=None, *args, **kwargs):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        if dataset_name == "FashionMNIST":
            self.train = datasets.FashionMNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transform
            )
            self.test = datasets.FashionMNIST(
                root=data_path,
                train=False,
                download=True,
                transform=transform
            )
        elif dataset_name == "MNIST":
            self.train = datasets.MNIST(
                root=data_path,
                train=True,
                download=True,
                transform=transform
            )
            self.test = datasets.MNIST(
                root=data_path,
                train=False,
                download=True,
                transform=transform
            )
        elif dataset_name == "CIFAR10":
            self.train = datasets.CIFAR10(
                root=data_path,
                train=True,
                download=True,
                transform=transform
            )
            self.test = datasets.CIFAR10(
                root=data_path,
                train=False,
                download=True,
                transform=transform
            )

        elif dataset_name == "SPOTS":

            data_loader = SPOT10Loader()
            images_train, targets_train = data_loader.get_data(dataset_dir=os.path.join('./data/SPOTS', "raw"), kind='train')
            images_test, targets_test = data_loader.get_data(dataset_dir=os.path.join('./data/SPOTS', "raw"), kind='test')

            self.train = CustomDataset(images_train, targets_train)
            self.test = CustomDataset(images_test, targets_test)
        else:
            raise ValueError(f"Dataset not supported: {dataset_name}")

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.data = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class EmbeddingDatasetWrapper:
    """
    Wrapper que expone .train y .test como datasets de embeddings y etiquetas.
    """

    def __init__(self, train, test, labels_train=None, labels_test=None):
        if isinstance(labels_train, np.ndarray):
            labels_train = torch.from_numpy(labels_train)
        if isinstance(labels_test, np.ndarray):
            labels_test = torch.from_numpy(labels_test)
        self.train = EmbeddingDataset(train, labels_train)
        self.test = EmbeddingDataset(test, labels_test)
        self.n_classes = len(torch.unique(torch.cat([labels_train, labels_test], dim=0)))