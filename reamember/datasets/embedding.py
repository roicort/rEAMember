import numpy as np
import torch
from torch.utils.data import Dataset

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
        self, train, test, labels_train=None, labels_test=None,
    ):
        if isinstance(labels_train, np.ndarray):
            labels_train = torch.from_numpy(labels_train)
        if isinstance(labels_test, np.ndarray):
            labels_test = torch.from_numpy(labels_test)
        
        self.train = EmbeddingDataset(train, labels_train)
        self.test = EmbeddingDataset(test, labels_test)
                                            
        if labels_train is not None and labels_test is not None:
            self.n_classes = len(
                torch.unique(torch.cat([labels_train, labels_test], dim=0))
            )
        else:
            self.n_classes = None
