import sys
import argparse
import click
from omegaconf import OmegaConf
from pathlib import Path
import torch

from reamember.config import setConfig

device = setConfig()

@click.group()
def cli():
    click.echo(f"[INFO] Using device: {device}")
    pass

@cli.command()
@click.option('--config', default='./config/256.yml', help='YAML configuration.')
def train_autoencoder(config):
    "Train autoencoder."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)
    if not path.exists():
        click.echo(f"[INFO] Creating path: {path}")
        path.mkdir(parents=True, exist_ok=True)

    # Load Dataset from Defaults
    if cfg.app.dataset == 'Custom':
        # For now, we will just print an error message
        # and exit since custom dataset implementation is not provided.
        # You can replace this with your actual dataset loading code.
        click.echo("[ERROR] Custom dataset not implemented yet.")
        sys.exit(1)
    else:
        click.echo(f"[INFO] Loading default image dataset: {cfg.app.dataset}")
        from reamember.dataset import ImageDatasetWrapper
        dataset = ImageDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

    input_shape = dataset.train[0][0].shape
    click.echo(f"[INFO] Input shape: {input_shape}")

    from reamember.train import train_autoencoder

    train_autoencoder(
        config=cfg.neural,
        input_shape=input_shape,
        dataset=dataset,
        name=f"{cfg.app.dataset}-{cfg.neural.latent_dim}",
        save_path=path / "autoencoder.pth"
    )

@cli.command()
@click.option('--config', default='./config/256.yml', help='YAML configuration.')
def get_embeddings(config):
    "Get embeddings from the encoder."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    # Load Dataset
    from reamember.dataset import ImageDatasetWrapper
    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = ImageDatasetWrapper(
        dataset_name=cfg.app.dataset,
    )

    input_shape = dataset.train[0][0].shape

    # Load Autoencoder
    from reamember.neuralnets.autoencoder import Autoencoder
    encoder = Autoencoder(input_shape=input_shape, latent_dim=cfg.neural.latent_dim)
    encoder_path = path / "autoencoder.pth"
    if encoder_path.exists():
        click.echo(f"[INFO] Loading encoder from: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    else:
        click.echo(f"[ERROR] Encoder path does not exist: {encoder_path}")
        sys.exit(1)

    from reamember.embeddings import get_embeddings

    get_embeddings(encoder, dataset, device=device, save_path=path)


@cli.command()
@click.option('--config', default='./config/256.yml', help='YAML configuration.')
def train_classifier(config):
    "Train classifier."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    from reamember.dataset import EmbeddingDatasetWrapper

    embeddings_dataset = torch.load(path / "embeddings.pth", map_location=device, weights_only=False)

    from reamember.train import train_classifier

    train_classifier(
        config=cfg.neural,
        dataset=embeddings_dataset,
        name=f"{cfg.app.dataset}-{cfg.neural.latent_dim}",
        save_path=path / "classifier.pth"
    )

if __name__ == "__main__":
    cli()

