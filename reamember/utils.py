import torch
import rich_click as click
from contextlib import contextmanager
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from omegaconf import ListConfig, OmegaConf
from collections.abc import Mapping
from pathlib import Path
import json
import sys
from tqdm import tqdm
import numpy as np
from reamember.eam.associative import NumpyAssociativeMemory as AssociativeMemory


console = Console()

# Configure rich click for better CLI output
rich_conf = click.RichHelpConfiguration(
    style_option="bold cyan",
    style_argument="bold cyan",
    style_command="bold cyan",
    style_switch="bold green",
    style_metavar="bold yellow",
    style_metavar_separator="dim",
    style_usage="bold yellow",
    style_usage_command="bold",
    style_helptext_first_line="",
    style_helptext="dim",
    style_option_default="dim",
    style_required_short="red",
    style_required_long="dim red",
    style_options_panel_border="dim",
    style_commands_panel_border="dim",
)

def config_summary(cfg):
    """
    Summarize the configuration.
    """
    # To python native types
    cfg_container = OmegaConf.to_container(cfg, resolve=True)

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.6g}"
        if value is None:
            return "null"
        if isinstance(value, (list, tuple)):
            simple = all(not isinstance(x, (dict, list, tuple)) for x in value)
            return (
                ", ".join(map(str, value))
                if len(value) <= 10 and simple
                else f"[{len(value)} items]"
            )
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def section_panel(title: str, data: dict) -> Panel:
        sub = Table(box=box.MINIMAL_HEAVY_HEAD)
        sub.add_column("Parameter", style="bold cyan")
        sub.add_column("Value", style="magenta")
        for k, v in data.items():
            sub.add_row(str(k), format_value(v))
        return Panel(sub, title=f"[bold]{title}[/bold]", border_style="dim")

    # Si existen secciones conocidas, mostrar subtables anidadas
    sections = [
        k
        for k in ("app", "neural", "memory")
        if k in cfg_container and isinstance(cfg_container[k], Mapping)
    ]

    if sections:
        panels = [section_panel(sec, cfg_container[sec]) for sec in sections]

        outer = Panel(
            Columns(panels, expand=True, equal=True),
            title="[bold]config[/bold]",
            border_style="cyan",
            padding=(0, 1),
            expand=True,
        )
        console.print(outer)


@contextmanager
def task_status(message: str, spinner: str = "dots"):
    """
    Show a Rich status spinner while a CLI task is running.
    """
    with console.status(f"[bold cyan]{message}[/bold cyan]", spinner=spinner):
        try:
            yield
        except Exception:
            console.print(f"[bold red]✗[/bold red] {message}")
            raise
        else:
            console.print(f"[bold green]✓[/bold green] {message}")


def fail_if_text_modality(cfg, command_name):
    """
    Abort early for commands that are not implemented for text modality yet.
    """
    if cfg.app.modality == "text":
        raise click.ClickException(
            f"El comando '{command_name}' no esta habilitado cuando app.modality es 'text'."
        )


def ensure_directory(path):
    """
    Create a directory if it does not exist yet.
    """
    path = Path(path)
    if not path.exists():
        click.echo(f"[INFO] Creating path: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_scalar_config_value(value):
    """
    Return the first item for list-like config values.
    """
    if isinstance(value, ListConfig):
        return value[0]
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def get_experiment_path(cfg, EXPERIMENTS_ROOT, latent=None):
    """
    Build the base experiment path for a dataset and optional latent size.
    """
    dataset_name = cfg.app.dataset.replace("/", "-")
    dataset_path = EXPERIMENTS_ROOT / dataset_name
    if cfg.app.modality == "text" and cfg.app.column:
        return dataset_path / f"{cfg.app.column}_{latent}"
    else:
        return dataset_path / f"latent_{latent}"


def load_embeddings_dataset(experiment_path, device):
    """
    Load the cached embeddings dataset from an experiment directory.
    """
    embeddings_path = Path(experiment_path) / "embeddings.pth"
    try:
        click.echo(f"[INFO] Loading embeddings dataset from: {embeddings_path}")
        return torch.load(embeddings_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        click.echo(f"[ERROR] Embeddings file not found: {embeddings_path}")
        click.echo("[INFO] Please run `get-embeddings` command first.")
        sys.exit(1)


def load_model_state(model, model_path, label, device, move_to_device=True):
    """
    Load a model checkpoint or abort with a consistent error message.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        click.echo(f"[ERROR] {label} path does not exist: {model_path}")
        sys.exit(1)

    click.echo(f"[INFO] Loading {label.lower()} from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    if move_to_device:
        model.to(device)
    return model


def decode_text_embeddings(model, embeddings, device=None, batch_size=32):
    """
    Decode embeddings into text batches using SONAR.
    """
    reconstructed_texts = []

    for start in tqdm(range(0, len(embeddings), batch_size), desc="Decoding texts"):
        batch = torch.as_tensor(
            embeddings[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            reconstructed_batch = model.decode(batch)

        if isinstance(reconstructed_batch, str):
            reconstructed_texts.append(reconstructed_batch)
        else:
            reconstructed_texts.extend(reconstructed_batch)

    return reconstructed_texts

class Quant:
    """Quantizer/dequantizer class.

    Provides the methods for quantizing and dequantizing data on the basis
    of an original corpus and a given number of discrete values. Quantization
    and dequantization is done using minima and maxima data per column.
    """

    def __init__(
        self,
        corpus: np.ndarray | torch.Tensor | None = None,
        *,
        per_dimension: bool = True,
        minima: np.ndarray | torch.Tensor | float | None = None,
        maxima: np.ndarray | torch.Tensor | float | None = None,
    ):
        self.per_dimension = per_dimension

        if minima is not None or maxima is not None:
            if minima is None or maxima is None:
                raise ValueError("Both minima and maxima must be provided together")
            self.minima = self._normalize_bound(minima)
            self.maxima = self._normalize_bound(maxima)
        else:
            if corpus is None:
                raise ValueError(
                    "Quant requires either a corpus or explicit minima and maxima"
                )
            self.minima, self.maxima = self.get_min_max(
                self._as_numpy(corpus),
                per_dimension=per_dimension,
            )

        if np.shape(self.minima) != np.shape(self.maxima):
            raise ValueError("Minima and maxima must have the same shape")

        idx = np.where(self.minima == self.maxima)[0]
        if len(idx) > 0:
            print(
                f'Minima and maxima have the same value in position(s): {idx.tolist()}'
            )

    @staticmethod
    def _as_numpy(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def from_global_bounds(cls, minima, maxima):
        return cls(minima=minima, maxima=maxima, per_dimension=False)

    @classmethod
    def from_bounds(cls, minima, maxima):
        minima = cls._normalize_bound(minima)
        maxima = cls._normalize_bound(maxima)
        return cls(
            minima=minima,
            maxima=maxima,
            per_dimension=np.ndim(minima) > 0,
        )

    @staticmethod
    def _normalize_bound(bound):
        bound = Quant._as_numpy(bound)
        if bound.ndim == 0:
            return float(bound)
        return np.ravel(bound).astype(float)

    def get_min_max(self, a: np.ndarray, per_dimension: bool = True):
        """Produces desirable minimum and maximum values for features."""
        if a.ndim == 0:
            raise ValueError("Corpus must have at least one dimension")
        if per_dimension:
            return np.min(a, axis=0), np.max(a, axis=0)
        return float(np.min(a)), float(np.max(a))

    def _iter_bounds(self, a: np.ndarray):
        values = self._as_numpy(a)
        if values.ndim != 1:
            raise ValueError(f'The array must have one dimension: {values.shape}.')

        if np.ndim(self.minima) == 0:
            return zip(
                values,
                np.repeat(self.minima, len(values)),
                np.repeat(self.maxima, len(values)),
            )

        if len(values) != len(self.minima):
            raise ValueError(
                "Input feature length does not match quantization bounds length"
            )
        return zip(values, self.minima, self.maxima)

    def quantize(self, a: np.ndarray, m: int):
        a = self._as_numpy(a)
        if a.ndim > 2:
            raise ValueError(f'The array as more than two dimensions: {a.shape}.')
        elif a.ndim == 1:
            b = [self._quantize(x, min, max, m) for x, min, max in self._iter_bounds(a)]
            return np.array(b, dtype=int)
        else:
            b = [self.quantize(e, m) for e in a]
            return np.array(b)

    def dequantize(self, a: np.array, m: int):
        a = self._as_numpy(a)
        if a.ndim > 2:
            raise ValueError(f'The array as more than two dimensions: {a.shape}.')
        elif a.ndim == 1:
            b = [
                self._dequantize(x, min, max, m) for x, min, max in self._iter_bounds(a)
            ]
            return np.array(b, dtype=float)
        else:
            b = [self.dequantize(e, m) for e in a]
            return np.array(b)

    def _quantize(self, x, min, max, m):
        if max == min:
            return round((m - 1) / 2)
        elif np.isnan(x):
            return max + 1
        else:
            return round((m - 1) * (x - min) / (max - min))

    def _dequantize(self, i, min, max, m):
        return (max - min) / 2 if m == 1 else (max - min) * i / (m - 1) + min