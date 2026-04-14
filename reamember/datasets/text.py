# Dataset y DataLoader para rEAMember
import torch
from torch.utils.data import Dataset
from datasets import load_dataset as load_dataset_hf

# Custom Dataset

class CustomTextDataset(Dataset):
    def __init__(
        self,
        texts,
        targets,
        transform=None,
        target_transform=torch.as_tensor,
    ):
        self.texts = texts
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.transform:
            text = self.transform(text)
        return text

def _identity(x):
    return x


def _clean_text(text):
    return "" if text is None else str(text).strip()


def _clean_definition(definition):
    definition = _clean_text(definition)
    if "." not in definition:
        return definition
    parts = [part.strip() for part in definition.split(".") if part.strip()]
    if not parts:
        return definition
    return max(parts, key=len)


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
            transform = _clean_definition if column == "definition" else _clean_text

        ds = load_dataset_hf(
            dataset_name, cache_dir=f"{data_path}/{dataset_name.replace('/', '_')}"
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
