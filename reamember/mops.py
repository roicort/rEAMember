import numpy as np
from tqdm import tqdm
from .eam.associative import AssociativeMemory

def memorize(cfg, dataset):

    features = dataset.train.data.cpu().numpy()

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
        device=dataset.train.data.device
        
    )

    for features in tqdm(features_rounded):
        eam.register(features)

    return eam

