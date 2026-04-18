##########################################################################################
# rEAMember                                                                              #
##########################################################################################

# A framework for experimenting with the EAM.

# --------------------------------------------------------------
# Base

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rich_click as click
import torch

from plotly import graph_objects as go

# --------------------------------------------------------------
# Rich

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from reamember.config import loadValConfig, setDeviceConfig

from reamember.datasets.image import ImageDatasetWrapper

# --------------------------------------------------------------
# EAM

from reamember.neuralnets.autoencoder import Autoencoder
from reamember.neuralnets.classifier import Classifier


from reamember.utils import (
    rich_conf,
    config_summary,
    ensure_directory,
    fail_if_text_modality,
    get_experiment_path,
    load_embeddings_dataset,
    load_model_state,
    task_status,
)

from reamember.pipes.image import (
    create_image_memories,
    get_bestimage_params,
    plot_image_memory_results,
    run_image_dream,
    test_image_encoder,
)
from reamember.pipes.text import (
    create_text_memories,
    get_besttext_params,
    get_text_embeddings,
    test_text_encoder,
)

##########################################################################################
# Config
##########################################################################################

# --------------------------------------------------------------
# Set the device for PyTorch

# This will automatically select GPU if available, otherwise CPU
device = setDeviceConfig()
EXPERIMENTS_ROOT = Path("experiments")
LOGS_PATH = Path("logs")


def load_cli_config(config):
    try:
        return loadValConfig(config)
    except Exception as exc:
        raise click.ClickException(
            f"Invalid configuration '{config}': {exc}"
        ) from exc

############################################################################################
# CLI Commands
############################################################################################


# Create a Click command group
@click.group()
@click.rich_config(help_config=rich_conf)
def cli():
    click.echo(f"[INFO] Using device: {device}")
    pass


@click.group()
def encoder():
    """Encoder related commands."""
    pass


@click.group()
def classifier():
    """Classifier related commands."""
    pass

# --------------------------------------------------------------
# Encoder Commands


@encoder.command(name="train")
@click.option("--config", help="YAML configuration.")
def train_encoder(config):
    "🏃🏻‍♂️‍➡️ Train autoencoder."
    cfg = load_cli_config(config)
    config_summary(cfg)

    if cfg.app.modality == "image":
        with task_status("Training image encoder"):
            from reamember.neuralnets.train import train_autoencoder

            dataset = ImageDatasetWrapper(
                dataset_name=cfg.app.dataset,
            )

            input_shape = dataset.train[0][0].shape
            click.echo(f"[INFO] Input shape: {input_shape}")

            for latent in cfg.neural.latent_dim:
                path = ensure_directory(get_experiment_path(cfg, EXPERIMENTS_ROOT, latent))

                train_autoencoder(
                    config=cfg.neural,
                    dim=latent,
                    input_shape=input_shape,
                    dataset=dataset,
                    name=f"{cfg.app.dataset}-{latent}",
                    save_path=path / "autoencoder.pth",
                )

    elif cfg.app.modality == "text":
        raise NotImplementedError(
            "Fine tuning SONAR is not implemented yet. Please use the pretrained model."
        )


@encoder.command(name="test")
@click.option("--config", help="YAML configuration.")
@click.option(
    "--n", default=0, help="Number of samples to reconstruct. If 0, reconstruct all."
)
def test_encoder(config, n):
    "Test encoder."
    cfg = load_cli_config(config)
    config_summary(cfg)

    if cfg.app.modality == "image":
        with task_status("Testing image encoder"):
            test_image_encoder(
                cfg,
                n=n,
                device=device,
                experiments_root=EXPERIMENTS_ROOT,
            )

    elif cfg.app.modality == "text":
        with task_status("Testing text encoder"):
            test_text_encoder(
                cfg,
                n_examples=n,
                device=device,
                experiments_root=EXPERIMENTS_ROOT,
            )

    # Done
    click.echo("[INFO] Encoder testing completed.")


# --------------------------------------------------------------
# Embedding Commands


@cli.command()
@click.option("--config", help="YAML configuration.")
def get_embeddings(config):
    "📊 Obtain embeddings from the encoder."

    from reamember.embeddings import get_embeddings

    cfg = load_cli_config(config)
    config_summary(cfg)

    # Load Dataset
    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    if cfg.app.modality == "image":
        with task_status("Generating image embeddings"):
            dataset = ImageDatasetWrapper(
                dataset_name=cfg.app.dataset,
            )

            input_shape = dataset.train[0][0].shape

            for latent in cfg.neural.latent_dim:
                path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)

                encoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
                encoder_path = path / "autoencoder.pth"
                load_model_state(
                    encoder,
                    encoder_path,
                    "Encoder",
                    device=device,
                    move_to_device=False,
                )

                get_embeddings(encoder, dataset, device=device, save_path=path)

    elif cfg.app.modality == "text":
        with task_status("Generating text embeddings"):
            get_text_embeddings(cfg, device=device, experiments_root=EXPERIMENTS_ROOT)

    # Done
    click.echo("[INFO] Embeddings obtained.")


# --------------------------------------------------------------
# Classifier Commands


@classifier.command(name="train")
@click.option("--config", help="YAML configuration.")
def train_classifier_command(config):
    "🏃🏻‍♂️‍➡️ Train classifier."

    from reamember.neuralnets.train import train_classifier

    cfg = load_cli_config(config)
    fail_if_text_modality(cfg, "classifier train")
    config_summary(cfg)

    with task_status("Training classifier"):
        for latent in cfg.neural.latent_dim:
            path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
            embeddings_dataset = load_embeddings_dataset(path, device)

            train_classifier(
                config=cfg.neural,
                dim=latent,
                dataset=embeddings_dataset,
                name=f"{cfg.app.dataset}-{latent}",
                save_path=path / "classifier.pth",
            )

    click.echo("[INFO] Classifier training completed.")


@classifier.command(name="test")
@click.option("--config", help="YAML configuration.")
def test_classifier_command(config):
    "👨🏻‍🏫 Test the classifier on the test set."
    cfg = load_cli_config(config)
    fail_if_text_modality(cfg, "classifier test")
    config_summary(cfg)

    with task_status("Testing classifier"):
        for latent in cfg.neural.latent_dim:
            path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
            embeddings_dataset = load_embeddings_dataset(path, device)

            classifier = Classifier(
                latent_dim=latent,
                n_classes=embeddings_dataset.n_classes,
            )

            classifier_path = path / "classifier.pth"
            load_model_state(classifier, classifier_path, "Classifier", device=device)

            predictions = []

            for f in tqdm(embeddings_dataset.test.data):
                f = torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
                predictions.append(classifier.predict(f).cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            targets = embeddings_dataset.test.targets.cpu().numpy()

            accuracy = accuracy_score(targets, predictions)
            report = classification_report(targets, predictions, output_dict=True)

            click.echo(f"[INFO] {latent} Accuracy: {accuracy}")
            click.echo(f"[INFO] {latent} Classification Report: {report}")

            report_path = path / "classifier_report.json"
            click.echo(f"[INFO] Saving classification report to: {report_path}")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)

            cm = confusion_matrix(targets, predictions)
            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=list(range(embeddings_dataset.n_classes)),
                    y=list(range(embeddings_dataset.n_classes)),
                    colorscale="Viridis",
                    colorbar=dict(title="Count"),
                )
            )
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Class",
                yaxis_title="True Class",
                width=800,
                height=800,
            )

            fig.write_image(path / "classifier_confmatrix.png")
            fig.write_html(path / "classifier_confmatrix.html")
            click.echo(f"[INFO] Saving confusion matrix to: {path / 'classifier_confmatrix.png'} and {path / 'classifier_confmatrix.html'}")

    click.echo("[INFO] Classifier testing completed.")


# --------------------------------------------------------------
# Memory Commands


@cli.command()
@click.option("--config", help="YAML configuration.")
def get_bestparams(config):
    "🔍 Search best memory sizes and filling percents."
    cfg = load_cli_config(config)
    config_summary(cfg)

    if cfg.app.modality == "text":
        get_besttext_params(
            cfg,
            config,
            device=device,
            EXPERIMENTS_ROOT=EXPERIMENTS_ROOT,
        )
        return

    get_bestimage_params(
        cfg,
        config,
        device=device,
        experiments_root=EXPERIMENTS_ROOT,
    )


@cli.command()
@click.option("--config", help="YAML configuration.")
@click.option("--n", default=0, help="Number of samples to recall. If 0, recall all.")
def create_memories(config, n):
    "🧠 Create memories."

    cfg = load_cli_config(config)
    config_summary(cfg)

    if cfg.app.modality == "image":
        with task_status("Creating image memories"):
            create_image_memories(
                cfg,
                n=n,
                device=device,
                experiments_root=EXPERIMENTS_ROOT,
            )
    elif cfg.app.modality == "text":
        with task_status("Creating text memories"):
            create_text_memories(
                cfg,
                n=n,
                device=device,
                experiments_root=EXPERIMENTS_ROOT,
            )


@cli.command()
@click.option("--config", help="YAML configuration.")
@click.option("--num-cycles", default=6, help="Número de ciclos de sueño.")
@click.option(
    "--init-type",
    default="real",
    type=click.Choice(["real", "random", "noisy"]),
    help="Tipo de vector inicial.",
)
@click.option(
    "--idx", default=0, help="Índice del vector inicial si es real o ruidoso."
)
def dream(config, num_cycles, init_type, idx):
    """
    🛌 Ejecuta el proceso de 'soñar' (dreaming) usando la memoria asociativa y el decoder.
    """

    cfg = load_cli_config(config)
    fail_if_text_modality(cfg, "dream")
    config_summary(cfg)
    with task_status("Running dream cycles"):
        run_image_dream(
            cfg,
            num_cycles=num_cycles,
            init_type=init_type,
            idx=idx,
            device=device,
            experiments_root=EXPERIMENTS_ROOT,
        )


# --------------------------------------------------------------
# Utils Commands


@cli.command()
@click.option("--config", help="YAML configuration.")
def plot(config):
    # Plots originales de WEAM
    cfg = load_cli_config(config)
    with task_status("Rendering memory plots"):
        plot_image_memory_results(cfg, experiments_root=EXPERIMENTS_ROOT)


@cli.command()
def launch_tensorboard():
    "Launch TensorBoard for monitoring."
    try:
        import subprocess

        with task_status("Launching TensorBoard"):
            subprocess.run(
                ["tensorboard", f"--logdir={Path('logs')}", "--port=6006", "--bind_all"],
                check=True,
            )
    except FileNotFoundError:
        click.echo("[ERROR] TensorBoard not found. Please install it.")
        sys.exit(1)


@cli.command()
def clean_logs():
    "Clean TensorBoard logs."
    with task_status("Cleaning logs"):
        log_path = LOGS_PATH
        if log_path.exists():
            shutil.rmtree(log_path)
            click.echo(f"[INFO] Logs cleaned: {log_path}")
        else:
            click.echo(f"[INFO] No logs to clean at: {log_path}")


#########################################################################
# Add commands and run the main CLI group

cli.add_command(encoder)
cli.add_command(classifier)

if __name__ == "__main__":
    cli()

#########################################################################