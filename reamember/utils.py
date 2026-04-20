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
from reamember.eam.associative import NumpyAssociativeMemory as AssociativeMemory


console = Console()

def create_associative_memory(cfg, latent, domain, device=None):
    """
    Create an associative memory instance from config values.
    """
    return AssociativeMemory(
        n=latent,
        m=domain,
        xi=cfg.memory.xi,
        sigma=cfg.memory.sigma,
        iota=cfg.memory.iota,
        kappa=cfg.memory.kappa,
        device=device,
    )

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

def _get_quantization_bounds(*feature_sets):
    joined = []
    for features in feature_sets:
        if features is None:
            continue
        joined.append(torch.as_tensor(features, dtype=torch.float32))

    if not joined:
        raise ValueError("At least one feature set is required to compute quantization bounds")

    all_features = torch.cat(joined, dim=0)
    return torch.min(all_features), torch.max(all_features)
