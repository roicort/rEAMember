import numpy as np
from tqdm import tqdm
from .eam.associative import AssociativeMemory

def rsize_recall(recall, msize, min_value, max_value):
    if (msize == 1):
        return (recall.astype(dtype=float) + 1.0)*(max_value - min_value)/2
    else:
        return (max_value - min_value) * recall.astype(dtype=float) \
            / (msize - 1.0) + min_value

def memorize(cfg, dataset):
    """
    Create and fill memory registering features from the dataset.
    """

    features = dataset.data.cpu().numpy()

    max_value = np.maximum(features, features)
    min_value = np.minimum(features, features)

    features_rounded = np.round(features, decimals=3)
    features_rounded = np.clip(features_rounded, min_value, max_value)

    eam = AssociativeMemory(
        n=cfg.memory.domain,
        m=cfg.neural.latent_dim,
        xi=cfg.memory.xi,
        sigma=cfg.memory.sigma,
        iota=cfg.memory.iota,
        kappa=cfg.memory.kappa,
        device=dataset.data.device
    )

    print(f"[INFO] Memorizing {len(features_rounded)} features with shape {features_rounded.shape}...")
    for features in tqdm(features_rounded):
        eam.register(features)

    return eam

def remember(cfg, eam, dataset):
    """
    Remember features from the dataset.
    """

    features = dataset.data.cpu().numpy()

    max_value = np.maximum(features, features)
    min_value = np.minimum(features, features)

    features_rounded = np.round(features, decimals=3)
    features_rounded = np.clip(features_rounded, min_value, max_value)

    memories_features = []
    memories_recognition = []
    memories_weights = []

    print(f"[INFO] Remembering {len(features_rounded)} features with shape {features_rounded.shape}...")
    for feature in tqdm(features):
        memory, recognized, weight = eam.recall(feature)
        memory = memory.cpu().numpy()
        recognized = recognized.cpu().numpy()
        weight = weight.cpu().numpy()

        memories_features.append(memory)
        memories_recognition.append(recognized)
        memories_weights.append(weight)

    memories_features = np.array(memories_features, dtype=float)
    memories_features = rsize_recall(memories_features, cfg.neural.latent_dim, min_value, max_value)
    memories_recognition = np.array(memories_recognition, dtype=int)
    memories_weights = np.array(memories_weights, dtype=float)

    return memories_features, memories_recognition, memories_weights