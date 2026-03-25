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
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import rich_click as click
import torch
import torchvision
from nltk.metrics import edit_distance
from omegaconf import OmegaConf
from plotly import graph_objects as go

# --------------------------------------------------------------
# Rich

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from reamember.config import loadValConfig, setDeviceConfig
from reamember.dataset import ImageDatasetWrapper, TextDatasetWrapper

# --------------------------------------------------------------
# EAM
# Can be changed to TorchAssociativeMemory or NumpyAssociativeMemory
from reamember.eam.associative import NumpyAssociativeMemory as AssociativeMemory
from reamember.mops import evalm, evalm_text, evalm_text_confusion, memorize, remember
from reamember.neuralnets.autoencoder import Autoencoder
from reamember.neuralnets.classifier import Classifier
from reamember.neuralnets.transformer import SONAR

from reamember.utils import (
    rich_conf,
    config_summary,
    ensure_directory,
    fail_if_text_modality,
    get_experiment_path,
    get_scalar_config_value,
    decode_text_embeddings,
    load_embeddings_dataset,
    load_model_state,
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

def create_sonar_model(runtime_device):
    """
    Build SONAR using MPS for encode when available and CPU for decode.
    """
    if getattr(runtime_device, "type", None) == "mps":
        click.echo(
            "[WARNING] SONAR decode is not supported on MPS. Using MPS for encode and CPU for decode."
        )
        return SONAR(
            encode_device=runtime_device,
            decode_device=torch.device("cpu"),
        )
    return SONAR(device=runtime_device)


def create_associative_memory(cfg, latent, domain):
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


def text_reconstruction_metrics(
    model, device, original_texts, embeddings, batch_size=32
):
    """
    Compute reconstruction metrics directly in embedding space and text space.
    """
    if len(original_texts) == 0:
        return [], {
            "samples": 0,
            "mean_cosine": 0.0,
            "mean_l2": 0.0,
            "mean_edit_distance": 0.0,
        }

    if getattr(device, "type", None) == "mps":
        device = torch.device("cpu")
        source_embeddings = torch.as_tensor(embeddings, dtype=torch.float32).cpu()
    else:
        source_embeddings = torch.as_tensor(embeddings, dtype=torch.float32).to(device)
    reconstructed_texts = decode_text_embeddings(
        model=model,
        embeddings=source_embeddings,
        device=device,
        batch_size=batch_size,
    )

    reconstructed_embeddings = []
    for start in tqdm(
        range(0, len(reconstructed_texts), batch_size),
        desc="Encoding reconstructed texts",
    ):
        batch_texts = reconstructed_texts[start : start + batch_size]
        with torch.no_grad():
            batch_embeddings = model.encode(batch_texts)
        reconstructed_embeddings.append(batch_embeddings.detach().cpu())

    reconstructed_embeddings = torch.cat(reconstructed_embeddings, dim=0)

    source_norm = torch.linalg.norm(source_embeddings, dim=1)
    reconstructed_norm = torch.linalg.norm(reconstructed_embeddings, dim=1)
    denom = torch.clamp(source_norm * reconstructed_norm, min=1e-12)
    cosine_scores = (
        torch.sum(source_embeddings * reconstructed_embeddings, dim=1) / denom
    )
    l2_scores = torch.linalg.norm(source_embeddings - reconstructed_embeddings, dim=1)

    samples = []
    for index, (original, reconstructed, cosine, l2) in enumerate(
        zip(original_texts, reconstructed_texts, cosine_scores, l2_scores)
    ):
        samples.append(
            {
                "index": index,
                "original": original,
                "reconstructed": reconstructed,
                "cosine": float(cosine.item()),
                "l2": float(l2.item()),
                "edit_distance": int(
                    edit_distance(original.lower(), reconstructed.lower())
                ),
            }
        )

    summary = {
        "samples": len(samples),
        "mean_cosine": float(np.mean([item["cosine"] for item in samples])),
        "mean_l2": float(np.mean([item["l2"] for item in samples])),
        "mean_edit_distance": float(
            np.mean([item["edit_distance"] for item in samples])
        ),
    }

    return samples, summary


def get_besttext_params(cfg, config):
    from sklearn.model_selection import KFold
    from reamember.dataset import EmbeddingDatasetWrapper

    global_results = []

    msizes = cfg.memory.domain
    filling_percents = [0.5]
    folds = cfg.memory.folds
    noise_level = cfg.memory.noise_level

    from rich.progress import (
        Progress,
        SpinnerColumn,
        TimeElapsedColumn,
        MofNCompleteColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        speed_estimate_period=5.0,
    )

    latent_task = progress.add_task(
        f"[magenta]Latent: {cfg.neural.latent_dim[0]}",
        total=len(cfg.neural.latent_dim),
        start=True,
    )
    msize_task = progress.add_task(
        f"[green]Memory Size: {msizes[0]}",
        total=len(msizes),
    )
    filling_task = progress.add_task(
        f"[blue]Filling Percent: {filling_percents[0]}",
        total=len(filling_percents),
    )
    fold_task = progress.add_task("[cyan]Folds", total=folds)

    progress.start()
    progress.start_task(latent_task)

    for latent in cfg.neural.latent_dim:
        path = get_experiment_path(cfg.app.dataset, EXPERIMENTS_ROOT, latent)
        embeddings_dataset = load_embeddings_dataset(path, device)

        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
            column=cfg.app.column,
        )

        transformer = create_sonar_model(device)

        X = embeddings_dataset.train.data
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        texts = np.array([str(text) for text in dataset.train.texts], dtype=object)

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        results = []

        progress.start_task(msize_task)
        progress.start_task(filling_task)
        progress.start_task(fold_task)

        for msize in msizes:
            for filling_percent in filling_percents:
                fold_metrics = []
                progress.reset(fold_task)

                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    texts_val = texts[val_idx].tolist()
                    split_index = max(1, len(X_train) // 2)

                    fold_train_wrapper = EmbeddingDatasetWrapper(
                        train=torch.tensor(X_train[:split_index], dtype=torch.float32),
                        test=torch.tensor(X_val, dtype=torch.float32),
                        noise_level=noise_level,
                    )

                    eam = create_associative_memory(cfg, latent, msize)

                    eam, _, _ = memorize(
                        eam,
                        dataset=fold_train_wrapper.train,
                        filling_percent=1.0,
                    )

                    (
                        memories_features,
                        recognitions,
                        _weights,
                        recognized,
                        unrecognized,
                    ) = evalm_text(eam, dataset=fold_train_wrapper.test)

                    recognized_embeddings = memories_features[recognitions]
                    recognized_texts = [
                        text
                        for text, is_recognized in zip(texts_val, recognitions)
                        if is_recognized
                    ]

                    _, summary = text_reconstruction_metrics(
                        model=transformer,
                        device=device,
                        original_texts=recognized_texts,
                        embeddings=recognized_embeddings,
                    )

                    fold_metrics.append(
                        {
                            "recognized": recognized,
                            "unrecognized": unrecognized,
                            "mean_cosine": summary["mean_cosine"],
                            "mean_l2": summary["mean_l2"],
                            "mean_edit_distance": summary["mean_edit_distance"],
                        }
                    )
                    progress.update(fold_task, advance=1)

                avg_metrics = {
                    key: np.mean([fold_metric[key] for fold_metric in fold_metrics])
                    for key in fold_metrics[0]
                }
                results.append(
                    {
                        "latent": latent,
                        "msize": msize,
                        "filling_percent": filling_percent,
                        **avg_metrics,
                    }
                )
                progress.update(
                    filling_task,
                    advance=1,
                    description=f"[blue]Filling Percent: {filling_percent}",
                )

            progress.reset(filling_task)
            progress.update(
                msize_task,
                advance=1,
                description=f"[green]Memory Size: {msize}",
            )

        progress.reset(msize_task)
        global_results.extend(results)
        progress.update(
            latent_task, advance=1, description=f"[magenta]Latent: {latent}"
        )

    progress.stop()

    path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
    save_path = path / "memories_results.json"
    click.echo(f"[INFO] Saving results to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(global_results, f, indent=4)

    df = pd.DataFrame(global_results)
    df = df.sort_values(
        by=["recognized", "mean_cosine", "mean_edit_distance", "mean_l2"],
        ascending=[False, False, True, True],
    )

    best_params = df.iloc[0].to_dict()
    cfg.neural.latent_dim = [int(best_params["latent"])]
    cfg.memory.domain = [int(best_params["msize"])]
    cfg.memory.filling = [float(best_params["filling_percent"])]

    config_path = Path(config)
    best_config_path = config_path.with_name(
        re.sub(r"\.yml$", ".best.yml", config_path.name)
    )
    click.echo(
        f"[INFO] Saving updated config with best parameters to: {best_config_path}"
    )
    with open(best_config_path, "w") as f:
        OmegaConf.save(cfg, f)

    click.echo("[INFO] Best text parameters search completed.")


# --------------------------------------------------------------
# Encoder Commands


@encoder.command(name="train")
@click.option("--config", help="YAML configuration.")
def train_encoder(config):
    "🏃🏻‍♂️‍➡️ Train autoencoder."
    cfg = load_cli_config(config)
    config_summary(cfg)

    if cfg.app.modality == "image":
        from reamember.train import train_autoencoder

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
        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
            column=cfg.app.column,
        )

        model = SONAR()
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
            path = ensure_directory(get_experiment_path(cfg, EXPERIMENTS_ROOT, latent))

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
        from reamember.dataset import ImageDatasetWrapper

        click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

        dataset = ImageDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        input_shape = dataset.test[0][0].shape

        for latent in cfg.neural.latent_dim:
            path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)

            autoencoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
            encoder_path = path / "autoencoder.pth"

            load_model_state(autoencoder, encoder_path, "Encoder", device=device)
            embeddings_dataset = load_embeddings_dataset(path, device)
            reconstructedImgPath = ensure_directory(path / "reconstructed")

            for i in tqdm(
                range(len(embeddings_dataset.test.data))
                if n == 0
                else range(min(n, len(embeddings_dataset.test.data)))
            ):
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
        from reamember.dataset import EmbeddingDatasetWrapper, TextDatasetWrapper
        from omegaconf import ListConfig

        click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
            column=cfg.app.column,
        )

        for latent in cfg.neural.latent_dim:
            path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)

            transformer = create_sonar_model(device)
            embeddings_dataset = load_embeddings_dataset(path, device=device)
            reconstructedTextPath = ensure_directory(path / "reconstructed")

            total = len(embeddings_dataset.test.data)
            limit = total if n == 0 else min(n, total)

            original_texts = [str(text) for text in dataset.test.texts[:limit]]
            test_embeddings = embeddings_dataset.test.data[:limit]

            samples, summary = text_reconstruction_metrics(
                model=transformer,
                device=device,
                original_texts=original_texts,
                embeddings=test_embeddings,
            )

            with open(
                reconstructedTextPath / "reconstructed.txt", "w", encoding="utf-8"
            ) as f_out:
                for sample in samples:
                    f_out.write(sample["reconstructed"] + "\n")

            metrics_path = reconstructedTextPath / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f_out:
                json.dump(
                    {
                        "dataset": cfg.app.dataset,
                        "latent_dim": int(latent),
                        "summary": summary,
                        "samples": samples,
                    },
                    f_out,
                    indent=4,
                    ensure_ascii=False,
                )

            domains = (
                [int(domain) for domain in cfg.memory.domain]
                if isinstance(cfg.memory.domain, ListConfig)
                else [int(cfg.memory.domain)]
            )
            filling_percent = 0.5
            train_embeddings = embeddings_dataset.train.data
            split_index = max(1, len(train_embeddings) // 2)
            memory_wrapper = EmbeddingDatasetWrapper(
                train=train_embeddings[:split_index],
                test=train_embeddings[split_index:],
            )
            confusion_summaries = []

            from rich.progress import (
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TimeElapsedColumn,
            )

            domain_progress = Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
            )
            domain_task = domain_progress.add_task(
                "[cyan]Domains",
                total=len(domains),
            )

            with domain_progress:
                for domain in domains:
                    eam = create_associative_memory(cfg, latent, domain)

                    eam, _, _ = memorize(
                        eam,
                        dataset=memory_wrapper.train,
                        filling_percent=1.0,
                    )

                    confusion = evalm_text_confusion(
                        eam,
                        seen_dataset=memory_wrapper.train,
                        unseen_dataset=memory_wrapper.test,
                    )

                    confusion_payload = {
                        "dataset": cfg.app.dataset,
                        "latent_dim": int(latent),
                        "memory_domain": domain,
                        "filling_percent": filling_percent,
                        "seen_source": "first_half_of_train",
                        "unseen_source": "second_half_of_train",
                        "labels": confusion["labels"],
                        "matrix": confusion["matrix"].tolist(),
                        "counts": confusion["counts"],
                        "rates": confusion["rates"],
                    }
                    confusion_summaries.append(confusion_payload)

                    confusion_path = (
                        reconstructedTextPath / f"recognition_confusion_domain_{domain}.json"
                    )
                    with open(confusion_path, "w", encoding="utf-8") as f_out:
                        json.dump(
                            confusion_payload,
                            f_out,
                            indent=4,
                            ensure_ascii=False,
                        )

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=confusion["matrix"],
                            x=confusion["labels"]["columns"],
                            y=confusion["labels"]["rows"],
                            colorscale="Viridis",
                            colorbar=dict(title="Count"),
                            text=confusion["matrix"],
                            texttemplate="%{text}",
                        )
                    )
                    fig.update_layout(
                        title=f"Text Memory Recognition Confusion Matrix (m={domain})",
                        xaxis_title="Memory Decision",
                        yaxis_title="Sample Type",
                        width=800,
                        height=500,
                    )
                    confusion_fig_path = (
                        reconstructedTextPath / f"recognition_confusion_domain_{domain}.html"
                    )
                    fig.write_html(confusion_fig_path)

                    click.echo(
                        f"[INFO] Recognition rates (m={domain}) | seen recognized: {confusion['rates']['seen_recognized_rate']:.4f} "
                        f"| unseen recognized: {confusion['rates']['unseen_recognized_rate']:.4f}"
                    )
                    click.echo(f"[INFO] Recognition confusion saved to: {confusion_path}")
                    click.echo(f"[INFO] Recognition confusion plot saved to: {confusion_fig_path}")
                    domain_progress.update(domain_task, advance=1)

            confusion_summary_path = reconstructedTextPath / "recognition_confusion_summary.json"
            with open(confusion_summary_path, "w", encoding="utf-8") as f_out:
                json.dump(
                    confusion_summaries,
                    f_out,
                    indent=4,
                    ensure_ascii=False,
                )

            click.echo(
                f"[INFO] Reconstruction metrics | cosine: {summary['mean_cosine']:.4f} "
                f"| l2: {summary['mean_l2']:.4f} "
                f"| edit distance: {summary['mean_edit_distance']:.4f}"
            )

            click.echo(f"[INFO] Reconstructed texts saved to: {reconstructedTextPath}")
            click.echo(f"[INFO] Reconstruction metrics saved to: {metrics_path}")
            click.echo(f"[INFO] Recognition confusion summary saved to: {confusion_summary_path}")

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
        dataset = ImageDatasetWrapper(
            dataset_name=cfg.app.dataset,
        )

        input_shape = dataset.train[0][0].shape

        for latent in cfg.neural.latent_dim:
            path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)

            # Load Autoencoder

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
        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
            column=cfg.app.column,
        )

        for latent in cfg.neural.latent_dim:
            path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)

            model = create_sonar_model(device)

            get_embeddings(
                model,
                dataset,
                modality=cfg.app.modality,
                device=device,
                save_path=path,
            )

    # Done
    click.echo("[INFO] Embeddings obtained.")


# --------------------------------------------------------------
# Classifier Commands


@classifier.command(name="train")
@click.option("--config", help="YAML configuration.")
def train_classifier_command(config):
    "🏃🏻‍♂️‍➡️ Train classifier."

    from reamember.train import train_classifier

    cfg = load_cli_config(config)
    fail_if_text_modality(cfg, "classifier train")
    config_summary(cfg)

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
    cfg = load_cli_config(config)
    config_summary(cfg)

    if cfg.app.modality == "text":
        get_besttext_params(cfg, config)
        return

    from reamember.neuralnets.classifier import Classifier
    from sklearn.model_selection import StratifiedKFold
    from reamember.dataset import EmbeddingDatasetWrapper

    global_results = []

    msizes = cfg.memory.domain
    filling_percents = cfg.memory.filling
    folds = cfg.memory.folds
    noise_level = cfg.memory.noise_level

    from rich.progress import (
        Progress,
        SpinnerColumn,
        TimeElapsedColumn,
        MofNCompleteColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        speed_estimate_period=5.0,
    )

    latent_task = progress.add_task(
        f"[magenta]Latent: {cfg.neural.latent_dim[0]}",
        total=len(cfg.neural.latent_dim),
        start=True,
    )
    msize_task = progress.add_task(
        f"[green]Memory Size: {msizes[0]}", total=len(msizes)
    )
    filling_task = progress.add_task(
        f"[blue]Filling Percent: {filling_percents[0]}", total=len(filling_percents)
    )
    fold_task = progress.add_task("[cyan]Folds", total=folds)

    progress.start()

    progress.start_task(latent_task)

    for latent in cfg.neural.latent_dim:
        path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)

        # Grid search over the memory size (m) and the filling percent.

        # Dataset ------------------------------------------------------------

        embeddings_dataset = load_embeddings_dataset(path, device=device)

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
        load_model_state(classifier, classifier_path, "Classifier", device=device)

        # Decoder -----------------------------------------------------------------

        decoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
        decoder_path = path / "autoencoder.pth"
        load_model_state(decoder, decoder_path, "Decoder", device=device)

        # Search --------------------------------------------------------------------

        results = []

        progress.start_task(msize_task)
        progress.start_task(filling_task)
        progress.start_task(fold_task)

        for msize in msizes:
            for filling_percent in filling_percents:
                fold_metrics = []
                progress.reset(fold_task)
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
                    eam = create_associative_memory(cfg, latent, msize)

                    # Memorize the dataset
                    eam, _, _ = memorize(
                        eam,
                        dataset=fold_train_wrapper.train,
                        filling_percent=filling_percent,
                    )

                    percentages, recall, precision = evalm(
                        eam, classifier=classifier, dataset=fold_train_wrapper.test
                    )

                    fold_metrics.append(
                        {
                            "precision": precision,
                            "recall": recall,
                            "recognized": percentages[0],
                            "unrecognized": percentages[1],
                            "correct": percentages[2],
                            "incorrect": percentages[3],
                        }
                    )
                    progress.update(fold_task, advance=1)
                avg_metrics = {
                    k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]
                }
                results.append(
                    {
                        "latent": latent,
                        "msize": msize,
                        "filling_percent": filling_percent,
                        **avg_metrics,
                    }
                )
                progress.update(
                    filling_task,
                    advance=1,
                    description=f"[blue]Filling Percent: {filling_percent}",
                )
            progress.reset(filling_task)
            progress.update(
                msize_task, advance=1, description=f"[green]Memory Size: {msize}"
            )
        progress.reset(msize_task)
        global_results.extend(results)
        progress.update(
            latent_task, advance=1, description=f"[magenta]Latent: {latent}"
        )

    progress.stop()
    # .................................................................

    path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
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
    best_config_path = config_path.with_name(
        re.sub(r"\.yml$", ".best.yml", config_path.name)
    )
    click.echo(
        f"[INFO] Saving updated config with best parameters to: {best_config_path}"
    )
    with open(best_config_path, "w") as f:
        OmegaConf.save(cfg, f)

    click.echo("[INFO] Best parameters search completed.")


@cli.command()
@click.option("--config", help="YAML configuration.")
@click.option("--n", default=0, help="Number of samples to recall. If 0, recall all.")
def create_memories(config, n):
    "🧠 Create memories."

    from reamember.neuralnets.classifier import Classifier

    cfg = load_cli_config(config)
    fail_if_text_modality(cfg, "create-memories")
    config_summary(cfg)

    latent = int(get_scalar_config_value(cfg.neural.latent_dim))
    domain = int(get_scalar_config_value(cfg.memory.domain))

    print(f"[INFO] Creating memories with latent={latent}, domain={domain}")

    path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)

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

    eam = create_associative_memory(cfg, latent, domain)

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
    load_model_state(classifier, classifier_path, "Classifier", device=device)

    # Decoder -----------------------------------------------------------------

    decoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
    decoder_path = path / "autoencoder.pth"
    load_model_state(decoder, decoder_path, "Decoder", device=device)

    # Inference ---------------------------------------------------------------

    reconstructedImgPath = ensure_directory(path / f"dim{domain}_memory_reconstructed")

    click.echo("[INFO] Classifying memories...")

    memories_recognition = []

    for i in tqdm(range(len(memories_features))):
        f = torch.as_tensor(
            memories_features[i], dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            memories_recognition.append(classifier.predict(f).cpu().numpy())

    for i in tqdm(
        range(len(memories_features))
        if n == 0
        else range(min(n, len(memories_features)))
    ):
        f = torch.as_tensor(
            memories_features[i], dtype=torch.float32, device=device
        ).unsqueeze(0)
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
    fig_path = path / "memories_confmatrix.svg"
    click.echo(f"[INFO] Saving confusion matrix to: {fig_path}")
    fig.write_image(str(fig_path))

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

    cfg = load_cli_config(config)
    fail_if_text_modality(cfg, "dream")
    config_summary(cfg)

    latent = int(get_scalar_config_value(cfg.neural.latent_dim))
    path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)

    # Cargar embeddings
    embeddings_dataset = load_embeddings_dataset(path, device)
    embeddings = (
        embeddings_dataset.test.data
        if hasattr(embeddings_dataset, "test")
        else embeddings_dataset["test"]["data"]
    )

    # Cargar memoria asociativa
    mem_params = cfg.memory if hasattr(cfg, "memory") else {}
    n = latent
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
    load_model_state(decoder, decoder_path, "Decoder", device=device)
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

    dreams_path = ensure_directory(path / "dreams")

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
    cfg = load_cli_config(config)
    # config_summary(cfg)
    path = EXPERIMENTS_ROOT / cfg.app.dataset.replace("/", "-")
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
            # print(subset)
            # Bar plot of msize vs unrecognized, correct & incorrect
            fig1 = px.bar(
                subset,
                x=subset["msize"].astype(
                    str
                ),  # Asegura que solo aparecen los presentes
                y=["unrecognized", "correct", "incorrect"],
                color_discrete_map={
                    "unrecognized": "blue",
                    "correct": "green",
                    "incorrect": "red",
                },
                title=f"Filling Percent: {filling_percent}, Latent: {latent}",
            )
            fig1.update_xaxes(type="category")

            # Plot precision and recall vs msize
            fig2 = px.scatter(
                subset,
                x="msize",
                y=["precision", "recall"],
                color_discrete_map={
                    "precision": "orange",
                    "recall": "purple",
                },
                title=f"Precision and Recall vs Memory Size (Latent: {latent}, Filling: {filling_percent})",
            )
            for trace in fig2.data:
                trace.mode = "lines+markers"
            fig2.update_xaxes(type="category")
            fig2.update_yaxes(range=[0, 1])
            fig2.update_layout(
                xaxis_title="Memory Size (m)",
                yaxis_title="Value",
                legend_title="Metrics",
                width=900,
                height=600,
            )

            ensure_directory(path / "plots")

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