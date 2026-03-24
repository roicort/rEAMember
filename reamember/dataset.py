# Dataset y DataLoader para rEAMember
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset as load_dataset_hf

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/"))
sys.path.append(path)
from SPOTS.utils import SPOT10Loader

from datasets.features import Image as HFImage
from PIL import Image as PILImage

# Custom Dataset


class CustomImageDataset(Dataset):
    def __init__(
        self,
        data,
        targets,
        transform=transforms.ToTensor(),
        target_transform=torch.as_tensor,
    ):
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
        assert img.ndim == 3, f"Shape incorrecto: {img.shape}"
        return img, target
    
class HFDataProxy:
    """Module-level proxy that lazily pulls images from a HF dataset.

    Placing this at module level makes it picklable for use with
    DataLoader(num_workers>0) and multiprocessing.
    """

    def __init__(self, hf_ds):
        self.hf_ds = hf_ds

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item["image"]
        if isinstance(img, np.ndarray):
            img = PILImage.fromarray(img)
        return img


class HFImageDataset(CustomImageDataset):

    def __init__(self, hf_ds, label_col=None, label_map=None, transform=None, target_transform=torch.as_tensor):
        """Wrap a Hugging Face image Dataset but provide the same
        interface as `CustomImageDataset` without materializing all images.

        We create a lightweight proxy for `data` that returns raw images from
        the HF dataset on indexing, and precompute `targets` if possible.
        """
        self.hf_ds = hf_ds
        self.label_col = label_col
        self.label_map = label_map

        # Determine labels if possible
        labels = None
        if self.label_col is not None:
            labels_list = []
            for i in range(len(self.hf_ds)):
                it = self.hf_ds[i]
                t = it[self.label_col]
                if self.label_map is not None and isinstance(t, str):
                    t = self.label_map[t]
                try:
                    labels_list.append(int(t))
                except Exception:
                    labels_list.append(t)
            try:
                labels = torch.as_tensor(labels_list)
            except Exception:
                labels = labels_list

        data_proxy = HFDataProxy(self.hf_ds)

        super().__init__(data=data_proxy, targets=labels, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.hf_ds)

class CustomTextDataset(Dataset):
    def __init__(
        self,
        texts,
        targets,
        transform=transforms.ToTensor(),
        target_transform=torch.as_tensor,
    ):
        self.texts = texts
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, target = self.texts[idx], self.targets[idx]
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            target = self.target_transform(target)
        return text, target


class ImageDatasetWrapper:
    def __init__(
        self,
        dataset_name="FashionMNIST",
        data_path="./data",
        transform=None,
        custom_class=None,
        *args,
        **kwargs,
    ):
        from torchvision import datasets

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        if dataset_name == "FashionMNIST":
            self.train = datasets.FashionMNIST(
                root=data_path, train=True, download=True, transform=transform
            )
            self.test = datasets.FashionMNIST(
                root=data_path, train=False, download=True, transform=transform
            )
        elif dataset_name == "MNIST":
            self.train = datasets.MNIST(
                root=data_path, train=True, download=True, transform=transform
            )
            self.test = datasets.MNIST(
                root=data_path, train=False, download=True, transform=transform
            )
        elif dataset_name == "CIFAR10":
            self.train = datasets.CIFAR10(
                root=data_path, train=True, download=True, transform=transform
            )
            self.test = datasets.CIFAR10(
                root=data_path, train=False, download=True, transform=transform
            )

        elif dataset_name == "SPOTS":
            data_loader = SPOT10Loader()
            images_train, targets_train = data_loader.get_data(
                dataset_dir=os.path.join("./data/SPOTS", "raw"), kind="train"
            )
            images_test, targets_test = data_loader.get_data(
                dataset_dir=os.path.join("./data/SPOTS", "raw"), kind="test"
            )

            self.train = CustomImageDataset(images_train, targets_train)
            self.test = CustomImageDataset(images_test, targets_test)

        elif dataset_name == "WikiArt":

            ds = load_dataset_hf("Artificio/WikiArt")

            # If no test split, create one from train (default 10% test)
            if "test" not in ds:
                test_size = kwargs.get("test_size", 0.1)
                seed = kwargs.get("seed", 42)
                split = ds["train"].train_test_split(test_size=test_size, seed=seed)
                ds["train"] = split["train"]
                ds["test"] = split["test"]

            label_col = 'genre'

            # Build a mapping for string labels to integer indices (if needed)
            label_map = None
            if label_col:
                sample_label = ds["train"][label_col][0]
                if isinstance(sample_label, str):
                    unique = sorted(list(set(ds["train"][label_col])))
                    label_map = {c: i for i, c in enumerate(unique)}

            # Create PyTorch datasets that apply transforms on the fly (no full memory copy)
            self.train = HFImageDataset(
                ds["train"], label_col=label_col, label_map=label_map, transform=transform
            )
            self.test = HFImageDataset(
                ds["test"], label_col=label_col, label_map=label_map, transform=transform
            )

            # Expose number of classes if possible
            try:
                if label_map is not None:
                    self.n_classes = len(label_map)
                elif label_col is not None:
                    self.n_classes = len(set(ds["train"][label_col]))
                else:
                    self.n_classes = None
            except Exception:
                self.n_classes = None


        else:
            raise ValueError(f"Dataset not supported: {dataset_name}")


def _identity(x):
    return x


class TextDatasetWrapper:
    def __init__(
        self,
        dataset_name="npvinHnivqn/EnglishDictionary",
        column="word",
        data_path="./data",
        transform=None,
        custom_class=None,
        *args,
        **kwargs,
    ):
        if transform is None:
            transform = _identity

        ds = load_dataset_hf(
            dataset_name, cache_dir=f"{data_path}/{dataset_name}"
        )

        # Detect if no test split and create one from train (default 10% test)
        if "test" not in ds:
            test_size = kwargs.get("test_size", 0.1)
            seed = kwargs.get("seed", 42)
            split = ds["train"].train_test_split(test_size=test_size, seed=seed)
            ds["train"] = split["train"]
            ds["test"] = split["test"]

        train_texts = ds["train"][column]
        test_texts = ds["test"][column]

        self.train = CustomTextDataset(
            train_texts, None, transform=transform
        )
        self.test = CustomTextDataset(test_texts, None, transform=transform)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets, noise_level=0.0):
        self.data = embeddings
        self.targets = targets
        self.noise_level = noise_level

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emb = self.data[idx]
        target = self.targets[idx]
        if self.noise_level > 0:
            noise = torch.randn_like(emb) * self.noise_level
            emb = emb + noise
        return emb, target


class EmbeddingDatasetWrapper:
    """
    Wrapper que expone .train y .test como datasets de embeddings y etiquetas.
    """

    def __init__(
        self, train, test, labels_train=None, labels_test=None, noise_level=0.0
    ):
        if isinstance(labels_train, np.ndarray):
            labels_train = torch.from_numpy(labels_train)
        if isinstance(labels_test, np.ndarray):
            labels_test = torch.from_numpy(labels_test)
        self.train = EmbeddingDataset(train, labels_train, noise_level=noise_level)
        self.test = EmbeddingDataset(test, labels_test, noise_level=noise_level)
        self.n_classes = len(
            torch.unique(torch.cat([labels_train, labels_test], dim=0))
        )
