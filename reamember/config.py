import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from numbers import Real

import torch
from omegaconf import DictConfig, ListConfig, MISSING, OmegaConf
from tqdm import tqdm


class Modality(str, Enum):
    image = "image"
    text = "text"


@dataclass
class CrossValConfig:
    folds: int = 5
    seed: int = 42


@dataclass
class AppConfig:
    dataset: str = MISSING
    modality: Modality = MISSING
    column: str | None = None
    crossval: CrossValConfig = field(default_factory=CrossValConfig)


@dataclass
class NeuralConfig:
    latent_dim: list[int] = field(default_factory=list)
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    patience: int | None = None
    delta: float | None = None


@dataclass
class MemoryConfig:
    noise_level: float | None = 0.0
    domain: list[int] = field(default_factory=list)
    batch_size: int | None = None
    filling: list[float] = field(default_factory=list)
    iota: Any = 0.0
    kappa: Any = 0.0
    xi: Any = 0.0
    sigma: Any = 0.1
    m: int | None = None


@dataclass
class RuntimeConfig:
    app: AppConfig = field(default_factory=AppConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def _normalize_special_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_special_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_special_values(item) for item in value]
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return value


def _require_non_empty_int_list(values: list[int], field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError(f"{field_name} must contain only positive integers")


def _numeric_values(value: Any, field_name: str) -> list[float]:
    if isinstance(value, (list, tuple, ListConfig)):
        values = value
    else:
        values = [value]

    if not values:
        raise ValueError(f"{field_name} must contain at least one value")

    numeric_values = []
    for item in values:
        if not isinstance(item, Real):
            raise ValueError(f"{field_name} values must be numeric")
        numeric_values.append(float(item))

    return numeric_values


def _validate_runtime_config(cfg: DictConfig) -> None:
    dataset = cfg.app.dataset.strip()
    if not dataset:
        raise ValueError("app.dataset must not be empty")

    if cfg.app.modality == Modality.text and not (cfg.app.column and cfg.app.column.strip()):
        raise ValueError("app.column is required when app.modality is 'text'")

    _require_non_empty_int_list(list(cfg.neural.latent_dim), "neural.latent_dim")
    _require_non_empty_int_list(list(cfg.memory.domain), "memory.domain")

    if not list(cfg.memory.filling):
        raise ValueError("memory.filling must contain at least one value")
    if any(value <= 0 or value > 1 for value in cfg.memory.filling):
        raise ValueError("memory.filling values must be in the range (0, 1]")

    iota_values = _numeric_values(cfg.memory.iota, "memory.iota")
    kappa_values = _numeric_values(cfg.memory.kappa, "memory.kappa")
    xi_values = _numeric_values(cfg.memory.xi, "memory.xi")
    sigma_values = _numeric_values(cfg.memory.sigma, "memory.sigma")

    if cfg.app.crossval.folds <= 1:
        raise ValueError("app.crossval.folds must be greater than 1")
    if cfg.app.crossval.seed < 0:
        raise ValueError("app.crossval.seed must be greater than or equal to 0")
    if cfg.memory.noise_level is not None and cfg.memory.noise_level < 0:
        raise ValueError("memory.noise_level must be greater than or equal to 0")
    if cfg.memory.batch_size is not None and cfg.memory.batch_size <= 0:
        raise ValueError("memory.batch_size must be a positive integer when provided")
    if any(value < 0 for value in iota_values):
        raise ValueError("memory.iota values must be greater than or equal to 0")
    if any(value < 0 for value in kappa_values):
        raise ValueError("memory.kappa values must be greater than or equal to 0")
    if any(value < 0 for value in xi_values):
        raise ValueError("memory.xi values must be greater than or equal to 0")
    if any(value < 0 for value in sigma_values):
        raise ValueError("memory.sigma must be greater than or equal to 0")
    if cfg.memory.m is not None and cfg.memory.m <= 0:
        raise ValueError("memory.m must be a positive integer when provided")


def loadValConfig(config_path: str | os.PathLike[str]) -> DictConfig:
    schema = OmegaConf.structured(RuntimeConfig)
    raw_cfg = OmegaConf.load(Path(config_path))
    normalized_cfg = OmegaConf.create(
        _normalize_special_values(OmegaConf.to_container(raw_cfg, resolve=False))
    )
    cfg = OmegaConf.merge(schema, normalized_cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _validate_runtime_config(cfg)
    return cfg

def setDeviceConfig():
    """
    Configura el dispositivo para PyTorch y spaCy según la disponibilidad de hardware.
    Devuelve el dispositivo configurado.
    """
    if torch.backends.mps.is_available():
        # Usar MPS (Metal Performance Shaders) en macOS
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # Usar CUDA (GPU NVIDIA) si está disponible
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        # Usar CPU como fallback
        device = torch.device("cpu")
    
    # Configurar TQDM en pandas
    tqdm.pandas()

    try:
        torch.ones(1, device=device)
    except Exception as e:
        print("Error:", e)
        raise e
    
    return device