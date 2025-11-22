##########################################################################################
# rEAMember                                                                              #
##########################################################################################

# A framework for experimenting with the EAM.

# --------------------------------------------------------------
# Base

import re
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import rich_click as click
import torch
import torchvision
from omegaconf import OmegaConf
from plotly import graph_objects as go
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel

# --------------------------------------------------------------
# Rich
from rich.table import Table
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from reamember.config import setConfig
from reamember.dataset import ImageDatasetWrapper, TextDatasetWrapper

# --------------------------------------------------------------
# EAM
# Can be changed to TorchAssociativeMemory or NumpyAssociativeMemory
from reamember.eam.associative import NumpyAssociativeMemory as AssociativeMemory
from reamember.mops import evalm, memorize, remember
from reamember.neuralnets.autoencoder import Autoencoder
from reamember.neuralnets.classifier import Classifier
from reamember.neuralnets.transformer import Transformer

##########################################################################################
# Config
##########################################################################################

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

# --------------------------------------------------------------
# Set the device for PyTorch

# This will automatically select GPU if available, otherwise CPU
device = setConfig()

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
# Format Utils


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
        Console().print(outer)


# --------------------------------------------------------------
# Encoder Commands


@encoder.command()
@click.option("--config", help="YAML configuration.")
def train(config):
    "🏃🏻‍♂️‍➡️ Train autoencoder."
    cfg = OmegaConf.load(config)
    config_summary(cfg)

    if cfg.app.modality == "image":
        from reamember.train import train_autoencoder

        dataset = ImageDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        input_shape = dataset.train[0][0].shape
        click.echo(f"[INFO] Input shape: {input_shape}")

        for latent in cfg.neural.latent_dim:
            path = f"experiments/{cfg.app.dataset}/latent_{latent}"
            path = Path(path)
            if not path.exists():
                click.echo(f"[INFO] Creating path: {path}")
                path.mkdir(parents=True, exist_ok=True)

            train_autoencoder(
                config=cfg.neural,
                dim=latent,
                input_shape=input_shape,
                dataset=dataset,
                name=f"{cfg.app.dataset}-{latent}",
                save_path=path / "autoencoder.pth",
            )

    elif cfg.app.modality == "text":
        from reamember.neuralnets.transformer import Transformer
        from reamember.train import train_transformer

        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        model = Transformer(model_name="bert-base-uncased")
        # Update cfg.neural.latent_dim to match model latent_dim
        cfg.neural.latent_dim = [model.latent_dim]
        # Save updated config
        config_path = Path(config)
        click.echo(
            f"[INFO] Saving updated config with best parameters to: {config_path}"
        )
        with open(config_path, "w") as f:
            OmegaConf.save(cfg, f)

        for latent in cfg.neural.latent_dim:
            path = f"experiments/{cfg.app.dataset}/latent_{latent}"
            path = Path(path)
            if not path.exists():
                click.echo(f"[INFO] Creating path: {path}")
                path.mkdir(parents=True, exist_ok=True)

            train_transformer(
                config=cfg.neural,
                dataset=dataset,
                name=f"{cfg.app.dataset}-{latent}",
                save_path=path / "transformer.pth",
            )


@encoder.command()
@click.option("--config", help="YAML configuration.")
@click.option(
    "--n", default=0, help="Number of samples to reconstruct. If 0, reconstruct all."
)
def test(config, n):
    "Test encoder."
    cfg = OmegaConf.load(config)
    config_summary(cfg)

    if cfg.app.modality == "image":
        from reamember.dataset import ImageDatasetWrapper

        click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

        dataset = ImageDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        input_shape = dataset.test[0][0].shape

        for latent in cfg.neural.latent_dim:
            path = f"experiments/{cfg.app.dataset}/latent_{latent}"
            path = Path(path)

            autoencoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
            encoder_path = path / "autoencoder.pth"

            if encoder_path.exists():
                click.echo(f"[INFO] Loading encoder from: {encoder_path}")
                autoencoder.load_state_dict(
                    torch.load(encoder_path, map_location=device)
                )
                autoencoder.to(device)
            else:
                click.echo(f"[ERROR] Encoder does not exist in: {encoder_path}")
                sys.exit(1)

            try:
                embeddings_dataset = torch.load(
                    path / "embeddings.pth", map_location=device, weights_only=False
                )
            except FileNotFoundError:
                click.echo(
                    f"[ERROR] Embeddings file not found: {path / 'embeddings.pth'}"
                )
                click.echo("[INFO] Please run `get-embeddings` command first.")
                sys.exit(1)

            reconstructedImgPath = Path(path / "reconstructed")
            if not reconstructedImgPath.exists():
                click.echo(f"[INFO] Creating path: {reconstructedImgPath}")
                reconstructedImgPath.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(len(embeddings_dataset.test.data)) if n == 0 else range(min(n, len(embeddings_dataset.test.data)))):
                f = torch.as_tensor(
                    embeddings_dataset.test.data[i], dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    reconstructed = autoencoder.decode(f)
                    # Save image using torchvision.utils.save_image
                    # print(reconstructed)
                    torchvision.utils.save_image(
                        reconstructed, reconstructedImgPath / f"img_{i}.png"
                    )

            click.echo(f"[INFO] Reconstructed images saved to: {reconstructedImgPath}")

    elif cfg.app.modality == "text":
        from reamember.dataset import TextDatasetWrapper

        click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        input_shape = dataset.test[0][0].shape

        for latent in cfg.neural.latent_dim:
            path = f"experiments/{cfg.app.dataset}/latent_{latent}"
            path = Path(path)

            transformer = Transformer(input_shape=input_shape, latent_dim=latent)
            transformer_path = path / "transformer.pth"

            if transformer_path.exists():
                click.echo(f"[INFO] Loading transformer from: {transformer_path}")
                transformer.load_state_dict(
                    torch.load(transformer_path, map_location=device)
                )
                transformer.to(device)
            else:
                click.echo(f"[ERROR] Transformer does not exist in: {transformer_path}")
                sys.exit(1)

            try:
                embeddings_dataset = torch.load(
                    path / "embeddings.pth", map_location=device, weights_only=False
                )
            except FileNotFoundError:
                click.echo(
                    f"[ERROR] Embeddings file not found: {path / 'embeddings.pth'}"
                )
                click.echo("[INFO] Please run `get-embeddings` command first.")
                sys.exit(1)

            reconstructedTextPath = Path(path / "reconstructed")
            if not reconstructedTextPath.exists():
                click.echo(f"[INFO] Creating path: {reconstructedTextPath}")
                reconstructedTextPath.mkdir(parents=True, exist_ok=True)

            with open(
                reconstructedTextPath / "reconstructed.txt", "w", encoding="utf-8"
            ) as f_out:
                for i in tqdm(range(len(embeddings_dataset.test.data))):
                    f = torch.as_tensor(
                        embeddings_dataset.test.data[i],
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    with torch.no_grad():
                        reconstructed = transformer.decode(f)
                        f_out.write(reconstructed + "\n")

            click.echo(f"[INFO] Reconstructed texts saved to: {reconstructedTextPath}")

    # Done
    click.echo("[INFO] Encoder testing completed.")


# --------------------------------------------------------------
# Embedding Commands


@cli.command()
@click.option("--config", help="YAML configuration.")
def get_embeddings(config):
    "📊 Obtain embeddings from the encoder."

    from reamember.embeddings import get_embeddings

    cfg = OmegaConf.load(config)
    config_summary(cfg)

    # Load Dataset
    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    if cfg.app.modality == "image":
        dataset = ImageDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        input_shape = dataset.train[0][0].shape

        for latent in cfg.neural.latent_dim:
            path = f"experiments/{cfg.app.dataset}/latent_{latent}"
            path = Path(path)

            # Load Autoencoder

            encoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
            encoder_path = path / "autoencoder.pth"
            if encoder_path.exists():
                click.echo(f"[INFO] Loading encoder from: {encoder_path}")
                encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            else:
                click.echo(f"[ERROR] Encoder path does not exist: {encoder_path}")
                sys.exit(1)

            get_embeddings(encoder, dataset, device=device, save_path=path)

    elif cfg.app.modality == "text":
        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        for latent in cfg.neural.latent_dim:
            path = f"experiments/{cfg.app.dataset}/latent_{latent}"
            path = Path(path)

            encoder_path = path / "transformer.pth"
            if encoder_path.exists():
                click.echo(f"[INFO] Loading transformer from: {encoder_path}")
                encoder = Transformer()
                encoder.load(encoder_path)
            else:
                click.echo(f"[ERROR] Transformer path does not exist: {encoder_path}")
                sys.exit(1)

            get_embeddings(
                encoder,
                dataset,
                modality=cfg.app.modality,
                device=device,
                save_path=path,
            )

    # Done
    click.echo("[INFO] Embeddings obtained.")


# --------------------------------------------------------------
# Classifier Commands


@classifier.command()
@click.option("--config", help="YAML configuration.")
def train(config):
    "🏃🏻‍♂️‍➡️ Train classifier."

    from reamember.train import train_classifier

    cfg = OmegaConf.load(config)
    config_summary(cfg)

    for latent in cfg.neural.latent_dim:
        path = f"experiments/{cfg.app.dataset}/latent_{latent}"
        path = Path(path)

        try:
            embeddings_dataset = torch.load(
                path / "embeddings.pth", map_location=device, weights_only=False
            )
        except FileNotFoundError:
            click.echo(f"[ERROR] Embeddings file not found: {path / 'embeddings.pth'}")
            click.echo("[INFO] Please run `get-embeddings` command first.")
            sys.exit(1)

        train_classifier(
            config=cfg.neural,
            dim=latent,
            dataset=embeddings_dataset,
            name=f"{cfg.app.dataset}-{latent}",
            save_path=path / "classifier.pth",
        )

    click.echo("[INFO] Classifier training completed.")


@classifier.command()
@click.option("--config", help="YAML configuration.")
def test(config):
    "👨🏻‍🏫 Test the classifier on the test set."
    cfg = OmegaConf.load(config)
    config_summary(cfg)

    for latent in cfg.neural.latent_dim:
        path = f"experiments/{cfg.app.dataset}/latent_{latent}"
        path = Path(path)

        embeddings_dataset = torch.load(
            path / "embeddings.pth", map_location=device, weights_only=False
        )

        classifier = Classifier(
            latent_dim=latent,
            n_classes=embeddings_dataset.n_classes,
        )

        classifier_path = path / "classifier.pth"
        if classifier_path.exists():
            click.echo(f"[INFO] Loading classifier from: {classifier_path}")
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
            classifier.to(device)
        else:
            click.echo(f"[ERROR] Classifier path does not exist: {classifier_path}")
            sys.exit(1)

        predictions = []

        for f in tqdm(embeddings_dataset.test.data):
            f = torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
            predictions.append(classifier.predict(f).cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = embeddings_dataset.test.targets.cpu().numpy()

        # Metrics
        accuracy = accuracy_score(targets, predictions)
        report = classification_report(targets, predictions, output_dict=True)

        click.echo(f"[INFO] {latent} Accuracy: {accuracy}")
        click.echo(f"[INFO] {latent} Classification Report: {report}")

        # Save the classification report
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
        fig_path = path / "classifier_confmatrix.html"
        click.echo(f"[INFO] Saving confusion matrix to: {fig_path}")
        fig.write_html(fig_path)

    click.echo("[INFO] Classifier testing completed.")


# --------------------------------------------------------------
# Memory Commands


@cli.command()
@click.option("--config", help="YAML configuration.")
def get_bestparams(config):
    "🔍 Search best memory sizes and filling percents."
    cfg = OmegaConf.load(config)
    config_summary(cfg)

    from reamember.neuralnets.classifier import Classifier
    from sklearn.model_selection import StratifiedKFold
    from reamember.neuralnets.classifier import Classifier
    from reamember.dataset import CustomImageDataset, EmbeddingDatasetWrapper


    global_results = []

    msizes = cfg.memory.domain
    filling_percents = cfg.memory.filling
    folds = cfg.memory.folds
    noise_level = cfg.memory.noise_level

    for latent in cfg.neural.latent_dim:
        path = f"experiments/{cfg.app.dataset}/latent_{latent}"
        path = Path(path)

        # Grid search over the memory size (m) and the filling percent.

        # Dataset ------------------------------------------------------------

        try:
            click.echo(
                f"[INFO] Loading embeddings dataset from: {path / 'embeddings.pth'}"
            )
            embeddings_dataset = torch.load(
                path / "embeddings.pth", map_location=device, weights_only=False
            )
        except FileNotFoundError:
            click.echo(f"[ERROR] Embeddings file not found: {path / 'embeddings.pth'}")
            click.echo("[INFO] Please run `get-embeddings` command first.")
            sys.exit(1)

        dataset = ImageDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        input_shape = dataset.train[0][0].shape
        click.echo(f"[INFO] Input shape: {input_shape}")

        X = embeddings_dataset.train.data
        y = embeddings_dataset.train.targets
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        # Classifier ------------------------------------------------------------

        classifier = Classifier(
            latent_dim=latent,
            n_classes=embeddings_dataset.n_classes,
        )

        classifier_path = path / "classifier.pth"
        if classifier_path.exists():
            click.echo(f"[INFO] Loading classifier from: {classifier_path}")
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
            classifier.to(device)
        else:
            click.echo(f"[ERROR] Classifier path does not exist: {classifier_path}")
            sys.exit(1)

        # Decoder -----------------------------------------------------------------

        decoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
        decoder_path = path / "autoencoder.pth"
        if decoder_path.exists():
            click.echo(f"[INFO] Loading decoder from: {decoder_path}")
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            decoder.to(device)
        else:
            click.echo(f"[ERROR] Decoder path does not exist: {decoder_path}")
            sys.exit(1)

        # Search --------------------------------------------------------------------

        results = []

        for msize in tqdm(msizes):
            for filling_percent in tqdm(filling_percents):
                click.echo(
                    f"[INFO] Testing msize={msize}, filling_percent={filling_percent}"
                )

                fold_metrics = []
                for train_idx, val_idx in skf.split(X, y):

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    fold_train_wrapper = EmbeddingDatasetWrapper(
                        train=torch.tensor(X_train),
                        test=torch.tensor(X_val),
                        labels_train=torch.tensor(y_train),
                        labels_test=torch.tensor(y_val),
                        noise_level=noise_level,
                    )

                    # Create a new memory instance with the current parameters
                    eam = AssociativeMemory(
                        n=latent,
                        m=msize,
                        xi=cfg.memory.xi,
                        sigma=cfg.memory.sigma,
                        iota=cfg.memory.iota,
                        kappa=cfg.memory.kappa,
                        device=device,
                    )

                    # Memorize the dataset
                    eam, _, _ = memorize(
                        eam,
                        dataset=fold_train_wrapper.train,
                        filling_percent=filling_percent,
                    )

                    percentages, recall, precision = evalm(
                        eam, classifier=classifier, dataset=fold_train_wrapper.test
                    )

                    fold_metrics.append({
                        "precision": precision,
                        "recall": recall,
                        "recognized": percentages[0],
                        "unrecognized": percentages[1],
                        "correct": percentages[2],
                        "incorrect": percentages[3],
                    })

                avg_metrics = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
                results.append({
                    "latent": latent,
                    "msize": msize,
                    "filling_percent": filling_percent,
                    **avg_metrics
                })

        global_results.extend(results)

    # .................................................................

    path = Path(f"experiments/{cfg.app.dataset}")
    save_path = path / "memories_results.json"
    click.echo(f"[INFO] Saving results to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(global_results, f, indent=4)

    df = pd.DataFrame(global_results)
    # Order columns by best precision
    df = df.sort_values(by=["recognized", "precision"], ascending=False)
    # Update cfg with best parameters
    best_params = df.iloc[0].to_dict()
    cfg.neural.latent_dim = [int(best_params["latent"])]
    cfg.memory.domain = [int(best_params["msize"])]
    cfg.memory.filling = [float(best_params["filling_percent"])]

    # Save updated config
    config_path = Path(config)
    best_config_path = config_path.with_name(re.sub(r"\.yml$", ".best.yml", config_path.name))
    click.echo(f"[INFO] Saving updated config with best parameters to: {best_config_path}")
    with open(best_config_path, "w") as f:
        OmegaConf.save(cfg, f)

    click.echo("[INFO] Best parameters search completed.")


@cli.command()
@click.option("--config", help="YAML configuration.")
@click.option(
    "--n", default=0, help="Number of samples to recall. If 0, recall all."
)
def create_memories(config, n):
    "🧠 Create memories."

    from omegaconf import ListConfig

    from reamember.neuralnets.classifier import Classifier

    cfg = OmegaConf.load(config)
    config_summary(cfg)

    latent = (
        int(cfg.neural.latent_dim[0])
        if isinstance(cfg.neural.latent_dim, ListConfig)
        else int(cfg.neural.latent_dim)
    )
    domain = (
        int(cfg.memory.domain[0])
        if isinstance(cfg.memory.domain, ListConfig)
        else int(cfg.memory.domain)
    )

    print(f"[INFO] Creating memories with latent={latent}, domain={domain}")

    path = f"experiments/{cfg.app.dataset}/latent_{latent}"
    path = Path(path)

    # Dataset ------------------------------------------------------------

    embeddings_dataset = torch.load(
        path / "embeddings.pth", map_location=device, weights_only=False
    )

    dataset = ImageDatasetWrapper(
        dataset_name=cfg.app.dataset,
    )

    input_shape = dataset.train[0][0].shape
    click.echo(f"[INFO] Input shape: {input_shape}")

    # Memory ---------------------------------------------------------------

    eam = AssociativeMemory(
        n=latent,
        m=domain,
        xi=cfg.memory.xi,
        sigma=cfg.memory.sigma,
        iota=cfg.memory.iota,
        kappa=cfg.memory.kappa,
        device=device,
    )

    eam, min_value, max_value = memorize(eam, dataset=embeddings_dataset.train)

    memories_features, memories_recognition, _ = remember(
        cfg,
        eam=eam,
        dataset=embeddings_dataset.test,
        min_value=min_value,
        max_value=max_value,
    )

    print("Memory features:", memories_features)
    print("Memory recognition:", memories_recognition)

    # Classifier ------------------------------------------------------------

    classifier = Classifier(
        latent_dim=latent,
        n_classes=embeddings_dataset.n_classes,
    )

    classifier_path = path / "classifier.pth"
    if classifier_path.exists():
        click.echo(f"[INFO] Loading classifier from: {classifier_path}")
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.to(device)
    else:
        click.echo(f"[ERROR] Classifier path does not exist: {classifier_path}")
        sys.exit(1)

    # Decoder -----------------------------------------------------------------

    decoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
    decoder_path = path / "autoencoder.pth"
    if decoder_path.exists():
        click.echo(f"[INFO] Loading decoder from: {decoder_path}")
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder.to(device)
    else:
        click.echo(f"[ERROR] Decoder path does not exist: {decoder_path}")
        sys.exit(1)

    # Inference ---------------------------------------------------------------

    reconstructedImgPath = Path(path / f"dim{domain}_memory_reconstructed")
    if not reconstructedImgPath.exists():
        click.echo(f"[INFO] Creating path: {reconstructedImgPath}")
        reconstructedImgPath.mkdir(parents=True, exist_ok=True)

    click.echo("[INFO] Classifying memories...")

    memories_recognition = []

    for i in tqdm(range(len(memories_features))) if n == 0 else range(min(n, len(memories_features))):
        f = torch.as_tensor(
            memories_features[i], dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            memories_recognition.append(classifier.predict(f).cpu().numpy())
            torchvision.utils.save_image(
                decoder.decode(f).cpu(), reconstructedImgPath / f"img_{i}.png"
            )  # Check

    # Logs --------------------------------------------------------------------

    memories_recognition = np.concatenate(memories_recognition, axis=0)
    original_labels = embeddings_dataset.test.targets.cpu().numpy()

    cm = confusion_matrix(original_labels, memories_recognition)
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
    fig_path = path / "memories_confmatrix.html"
    click.echo(f"[INFO] Saving confusion matrix to: {fig_path}")
    fig.write_html(fig_path)

    # click.echo(f"[INFO] Saving memories to: {path / 'memories.pth'}")
    # torch.save({
    #    'features': memories_features,
    #    'recognition': memories_recognition,
    #    'weights': memories_weights
    # }, path / "memories.pth")


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

    cfg = OmegaConf.load(config)
    config_summary(cfg)

    path = Path(f"experiments/{cfg.app.dataset}/latent_{cfg.neural.latent_dim}")

    # Cargar embeddings
    embeddings_path = path / "embeddings.pth"
    if not embeddings_path.exists():
        click.echo(f"[ERROR] Embeddings file not found: {embeddings_path}")
        return
    embeddings_dataset = torch.load(embeddings_path, map_location=device)
    embeddings = (
        embeddings_dataset.test.data
        if hasattr(embeddings_dataset, "test")
        else embeddings_dataset["test"]["data"]
    )

    # Cargar memoria asociativa
    mem_params = cfg.memory if hasattr(cfg, "memory") else {}
    n = (
        cfg.neural.latent_dim[0]
        if isinstance(cfg.neural.latent_dim, (list, tuple))
        else cfg.neural.latent_dim
    )
    m = cfg.memory.m if hasattr(cfg.memory, "m") else 4
    memory = AssociativeMemory(n=n, m=m, device=device, **mem_params)
    # Llenar memoria con embeddings de entrenamiento
    train_embeddings = (
        embeddings_dataset.train.data
        if hasattr(embeddings_dataset, "train")
        else embeddings_dataset["train"]["data"]
    )
    for vec in train_embeddings:
        memory.register(vec.to(device))

    # Cargar decoder
    input_shape = (
        embeddings_dataset.input_shape
        if hasattr(embeddings_dataset, "input_shape")
        else (1, 28, 28)
    )
    decoder = Autoencoder(input_shape=input_shape, latent_dim=n)
    decoder_path = path / "autoencoder.pth"
    if not decoder_path.exists():
        click.echo(f"[ERROR] Decoder file not found: {decoder_path}")
        return
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.to(device)
    decoder.eval()

    # Selección del vector inicial
    if init_type == "real":
        vector = embeddings[idx].to(device)
    elif init_type == "random":
        vector = torch.randint(
            0, memory.m, (memory.n,), dtype=torch.int16, device=device
        )
    elif init_type == "noisy":
        vector = embeddings[idx].to(device)
        noise = torch.randn_like(vector) * 0.1
        vector = torch.clamp(vector + noise, 0, memory.m - 1).to(torch.int16)
    else:
        raise ValueError("init_type debe ser 'real', 'random' o 'noisy'")

    dreams_path = path / "dreams"
    dreams_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_cycles):
        recalled, accepted, weight = memory.recall(vector)
        # Decodifica (asegúrate de que recalled esté en el formato correcto para el decoder)
        decoded = decoder.decode(
            torch.tensor(recalled, dtype=torch.float32, device=device).unsqueeze(0)
        )
        torchvision.utils.save_image(decoded, dreams_path / f"dream_{i}.png")
        vector = torch.tensor(recalled, dtype=torch.float32, device=device)

    click.echo(f"[INFO] Sueños guardados en: {dreams_path}")


# --------------------------------------------------------------
# Utils Commands

@cli.command()
@click.option("--config", help="YAML configuration.")
def plot(config):
    # Plots originales de WEAM
    cfg = OmegaConf.load(config)
    #config_summary(cfg)
    path = Path(f"experiments/{cfg.app.dataset}")
    load_path = path / "memories_results.json"
    print(f"[INFO] Loading results from: {load_path}")

    with open(load_path) as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        print(df)
        filling_percent = cfg.memory.filling
        print(f"[INFO] Filtered results for filling percent '{filling_percent}':")
        newdf = df[df["filling_percent"] == filling_percent]

        # For every latent dimension, plot the results using plotly
        for latent in newdf["latent"].unique():
            subset = newdf[newdf["latent"] == latent]
            #print(subset)
            # Bar plot of msize vs unrecognized, correct & incorrect
            fig1 = px.bar(
                subset,
                x=subset["msize"].astype(str),  # Asegura que solo aparecen los presentes
                y=["unrecognized", "correct", "incorrect"],
                title=f"Filling Percent: {filling_percent}, Latent: {latent}"
            )
            fig1.update_xaxes(type='category')

            # Plot precision and recall vs msize
            fig2 = px.scatter(
                subset,
                x="msize",
                y=["precision", "recall"],
                title=f"Precision and Recall vs Memory Size (Latent: {latent}, Filling: {filling_percent})"
            )
            for trace in fig2.data:
                trace.mode = "lines+markers"
            fig2.update_xaxes(type='category')
            fig2.update_yaxes(range=[0, 1])
            fig2.update_layout(
                xaxis_title="Memory Size (m)",
                yaxis_title="Value",
                legend_title="Metrics",
                width=900,
                height=600,
            )

            save_path = path / "plots"
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)

            fig1.write_image(path / f"plots/memory_results_latent{latent}.svg")
            fig2.write_image(path / f"plots/memory_scores_latent{latent}.svg")

@cli.command()
def launch_tensorboard():
    "Launch TensorBoard for monitoring."
    click.echo("[INFO] Running TensorBoard...")
    try:
        import subprocess

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
    click.echo("[INFO] Cleaning logs...")
    log_path = Path("logs")
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