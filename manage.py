import sys
import torch
import shutil
import json
import torchvision
import pandas as pd
import numpy as np
import rich_click as click
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import plotly.express as px
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Can be changed to TorchAssociativeMemory or NumpyAssociativeMemory
from reamember.eam.associative import TorchAssociativeMemory as AssociativeMemory 

from reamember.neuralnets.classifier import Classifier
from reamember.dataset import ImageDatasetWrapper
from reamember.neuralnets.autoencoder import Autoencoder
from reamember.config import setConfig
from reamember.mops import memorize, evalm, remember

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

#--------------------------------------------------------------
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
def autoencoder():
    """Comandos para el autoencoder."""
    pass

@click.group()
def classifier():
    """Comandos para el clasificador."""
    pass

#--------------------------------------------------------------
# Autoencoder Commands

@autoencoder.command()
@click.option("--config", help="YAML configuration.")
def train(config):
    "🏃🏻‍♂️‍➡️ Train autoencoder."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)
    if not path.exists():
        click.echo(f"[INFO] Creating path: {path}")
        path.mkdir(parents=True, exist_ok=True)

    # Load Dataset from Defaults
    if cfg.app.dataset == "Custom":
        # For now, we will just print an error message
        # and exit since custom dataset implementation is not provided.
        # You can replace this with your actual dataset loading code.
        click.echo("[ERROR] Custom dataset not implemented yet.")
        sys.exit(1)
    else:
        click.echo(f"[INFO] Loading default image dataset: {cfg.app.dataset}")

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
        save_path=path / "autoencoder.pth",
    )

@autoencoder.command()
@click.option("--config", help="YAML configuration.")
def test(config):
    "Test autoencoder."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    from reamember.dataset import ImageDatasetWrapper

    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = ImageDatasetWrapper(
        dataset_name=cfg.app.dataset,
    )

    input_shape = dataset.test[0][0].shape

    autoencoder = Autoencoder(input_shape=input_shape, latent_dim=cfg.neural.latent_dim)
    encoder_path = path / "autoencoder.pth"
    if encoder_path.exists():
        click.echo(f"[INFO] Loading encoder from: {encoder_path}")
        autoencoder.load_state_dict(torch.load(encoder_path, map_location=device))
        autoencoder.to(device)
    else:
        click.echo(f"[ERROR] Encoder path does not exist: {encoder_path}")
        sys.exit(1)
    
    try:
        embeddings_dataset = torch.load(
            path / "embeddings.pth", map_location=device, weights_only=False
        )
    except FileNotFoundError:
        click.echo(f"[ERROR] Embeddings file not found: {path / 'embeddings.pth'}")
        click.echo("[INFO] Please run `get-embeddings` command first.")
        sys.exit(1)

    reconstructedImgPath = Path(path / "reconstructed")
    if not reconstructedImgPath.exists():
        click.echo(f"[INFO] Creating path: {reconstructedImgPath}")
        reconstructedImgPath.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(embeddings_dataset.test.data))):
        f = torch.as_tensor(embeddings_dataset.test.data[i], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            reconstructed = autoencoder.decode(f)
            # Save image using torchvision.utils.save_image
            torchvision.utils.save_image(reconstructed, reconstructedImgPath / f"img_{i}.png")

    click.echo(f"[INFO] Reconstructed images saved to: {reconstructedImgPath}")

#--------------------------------------------------------------
# Embedding Commands

@cli.command()
@click.option("--config", help="YAML configuration.")
def get_embeddings(config):
    "📊 Obtain embeddings from the encoder."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    # Load Dataset

    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = ImageDatasetWrapper(
        dataset_name=cfg.app.dataset,
    )

    input_shape = dataset.train[0][0].shape

    # Load Autoencoder

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

#--------------------------------------------------------------
# Classifier Commands

@classifier.command()
@click.option("--config", help="YAML configuration.")
def train(config):
    "🏃🏻‍♂️‍➡️ Train classifier."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    try:
        embeddings_dataset = torch.load(
            path / "embeddings.pth", map_location=device, weights_only=False
        )
    except FileNotFoundError:
        click.echo(f"[ERROR] Embeddings file not found: {path / 'embeddings.pth'}")
        click.echo("[INFO] Please run `get-embeddings` command first.")
        sys.exit(1)

    from reamember.train import train_classifier

    train_classifier(
        config=cfg.neural,
        dataset=embeddings_dataset,
        name=f"{cfg.app.dataset}-{cfg.neural.latent_dim}",
        save_path=path / "classifier.pth",
    )

@classifier.command()
@click.option("--config", help="YAML configuration.")
def test(config):
    "👨🏻‍🏫 Test the classifier on the test set."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    embeddings_dataset = torch.load(
        path / "embeddings.pth", map_location=device, weights_only=False
    )

    classifier = Classifier(
        latent_dim=cfg.neural.latent_dim,
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

    # Predict on the test set
    click.echo("[INFO] Testing classifier...")

    predictions = []

    for f in tqdm(embeddings_dataset.test.data):
        f = torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
        predictions.append(classifier.predict(f).cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = embeddings_dataset.test.targets.cpu().numpy()

    # Metrics
    accuracy = accuracy_score(targets, predictions)
    report = classification_report(targets, predictions, output_dict=True)

    click.echo(f"[INFO] Accuracy: {accuracy}")
    click.echo(f"[INFO] Classification Report: {report}")

    # Save the classification report
    report_path = path / "classification_report.json"
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

#--------------------------------------------------------------
# Memory Commands

@cli.command()
@click.option("--config", help="YAML configuration.")
def get_bestparams(config):
    "🔍 Search best memory sizes and filling percents."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    # Grid search over the memory size (m) and the filling percent.

    msizes = [1,2,4,8,16,32,64,128,256,512]
    filling_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Dataset ------------------------------------------------------------

    try:
        click.echo(f"[INFO] Loading embeddings dataset from: {path / 'embeddings.pth'}")
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


    # Classifier ------------------------------------------------------------

    from reamember.neuralnets.classifier import Classifier

    classifier = Classifier(
        latent_dim=cfg.neural.latent_dim,
        n_classes=embeddings_dataset.n_classes,
    )

    classifier_path = path / "classifier.pth"
    if classifier_path.exists():
        click.echo(f"[INFO] Loading classifier from: {classifier_path}")
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    else:
        click.echo(f"[ERROR] Classifier path does not exist: {classifier_path}")
        sys.exit(1)

    classifier.to(device)

    # Decoder -----------------------------------------------------------------

    decoder = Autoencoder(input_shape=input_shape, latent_dim=cfg.neural.latent_dim)
    decoder_path = path / "autoencoder.pth"
    if decoder_path.exists():
        click.echo(f"[INFO] Loading decoder from: {decoder_path}")
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    else:
        click.echo(f"[ERROR] Decoder path does not exist: {decoder_path}")
        sys.exit(1)
    decoder.to(device)

    # Search --------------------------------------------------------------------

    results = []

    for msize in tqdm(msizes):
        for filling_percent in tqdm(filling_percents):
            click.echo(f"[INFO] Testing msize={msize}, filling_percent={filling_percent}")

            # Create a new memory instance with the current parameters
            eam = AssociativeMemory(
                n=cfg.neural.latent_dim,
                m=msize,
                xi=cfg.memory.xi,
                sigma=cfg.memory.sigma,
                iota=cfg.memory.iota,
                kappa=cfg.memory.kappa,
                device=device
            )

            # Memorize the dataset
            eam = memorize(eam, dataset=embeddings_dataset.train, filling_percent=filling_percent)
            recognized, accuracy = evalm(eam, classifier=classifier, dataset=embeddings_dataset.test)

            results.append({
                "msize": msize,
                "filling_percent": filling_percent,
                "recognized": recognized,
                "accuracy": accuracy,
            })

    save_path = path / "bestparams_results.json"
    click.echo(f"[INFO] Saving results to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    df = pd.DataFrame(results)
    heatmap_data = df.pivot(index="filling_percent", columns="msize", values="accuracy")

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="msize", y="filling_percent", color="Accuracy"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(title="Accuracy según msize y filling_percent")
    fig.write_html(path / "bestparams_heatmap.html")
        

@cli.command()
@click.option("--config", help="YAML configuration.")
def create_memories(config):
    "🧠 Create memories."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
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
        n=cfg.neural.latent_dim,
        m=cfg.memory.domain,
        xi=cfg.memory.xi,
        sigma=cfg.memory.sigma,
        iota=cfg.memory.iota,
        kappa=cfg.memory.kappa,
        device=device
    )


    eam = memorize(eam, dataset=embeddings_dataset.test)
    
    memories_features, memories_recognition, memories_weights = remember(cfg,
        eam=eam,
        dataset=embeddings_dataset.test
    )

    # Classifier ------------------------------------------------------------

    from reamember.neuralnets.classifier import Classifier

    classifier = Classifier(
        latent_dim=cfg.neural.latent_dim,
        n_classes=embeddings_dataset.n_classes,
    )

    classifier_path = path / "classifier.pth"
    if classifier_path.exists():
        click.echo(f"[INFO] Loading classifier from: {classifier_path}")
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    else:
        click.echo(f"[ERROR] Classifier path does not exist: {classifier_path}")
        sys.exit(1)

    classifier.to(device)

    # Decoder -----------------------------------------------------------------

    decoder = Autoencoder(input_shape=input_shape, latent_dim=cfg.neural.latent_dim)
    decoder_path = path / "autoencoder.pth"
    if decoder_path.exists():
        click.echo(f"[INFO] Loading decoder from: {decoder_path}")
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    else:
        click.echo(f"[ERROR] Decoder path does not exist: {decoder_path}")
        sys.exit(1)
    decoder.to(device)

    # Inference ---------------------------------------------------------------

    reconstructedImgPath = Path(path / "memory_reconstructed")
    if not reconstructedImgPath.exists():
        click.echo(f"[INFO] Creating path: {reconstructedImgPath}")
        reconstructedImgPath.mkdir(parents=True, exist_ok=True)

    click.echo("[INFO] Classifying memories...")

    memories_recognition = []

    for i in tqdm(range(len(memories_features))):
        f = torch.as_tensor(memories_features[i], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            memories_recognition.append(
                classifier.predict(f).cpu().numpy()
            )
            torchvision.utils.save_image(decoder.decode(f).cpu(), reconstructedImgPath / f"img_{i}.png")

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
def dream(config):
    "🛌 Not implemented"
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)
    pass


#--------------------------------------------------------------
# Utils Commands

@cli.command()
def launch_tensorboard():
    "Launch TensorBoard for monitoring."
    click.echo("[INFO] Running TensorBoard...")
    try:
        import subprocess

        subprocess.run(
            ["tensorboard", f"--logdir={Path('logs')}", "--port=6006"], check=True
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

cli.add_command(autoencoder)
cli.add_command(classifier)

if __name__ == "__main__":
    cli()
