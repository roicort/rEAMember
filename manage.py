import sys
import torch
import shutil
import numpy as np
import rich_click as click
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    style_commands_panel_border="dim"
)

from reamember.config import setConfig

device = setConfig()

@click.group()
@click.rich_config(help_config=rich_conf)
def cli():
    click.echo(f"[INFO] Using device: {device}")
    pass

@cli.command()
@click.option('--config',  help='YAML configuration.')
def train_autoencoder(config):
    "🏃🏻‍♂️‍➡️ Train autoencoder."
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
@click.option('--config',  help='YAML configuration.')
def get_embeddings(config):
    "📊 Obtain embeddings from the encoder."
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
@click.option('--config',  help='YAML configuration.')
def train_classifier(config):
    "🏃🏻‍♂️‍➡️ Train classifier."
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

@cli.command()
@click.option('--config',  help='YAML configuration.')
def test_autoencoder(config):
    "⚠️ Not implemented"
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    from reamember.dataset import ImageDatasetWrapper
    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = ImageDatasetWrapper(
        dataset_name=cfg.app.dataset,
    )

    input_shape = dataset.train[0][0].shape

    from reamember.neuralnets.autoencoder import Autoencoder
    encoder = Autoencoder(input_shape=input_shape, latent_dim=cfg.neural.latent_dim)
    encoder_path = path / "autoencoder.pth"
    if encoder_path.exists():
        click.echo(f"[INFO] Loading encoder from: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    else:
        click.echo(f"[ERROR] Encoder path does not exist: {encoder_path}")
        sys.exit(1)
    pass

@cli.command()
@click.option('--config',  help='YAML configuration.')
def test_classifier(config):
    "👨🏻‍🏫 Test the classifier on the test set."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    from reamember.dataset import EmbeddingDatasetWrapper

    embeddings_dataset = torch.load(path / "embeddings.pth", map_location=device, weights_only=False)

    from reamember.neuralnets.classifier import Classifier

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
    
    from plotly import graph_objects as go

    cm = confusion_matrix(targets, predictions)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=list(range(embeddings_dataset.n_classes)),
        y=list(range(embeddings_dataset.n_classes)),
        colorscale='Viridis',
        colorbar=dict(title='Count')
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        width=800,
        height=800
    )
    fig_path = path / "classifier_confmat.html"
    click.echo(f"[INFO] Saving confusion matrix to: {fig_path}")
    fig.write_html(fig_path)

@cli.command()
@click.option('--config',  help='YAML configuration.')
def create_memories(config):
    "🧠 Create memories."
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)

    embeddings_dataset = torch.load(path / "embeddings.pth", map_location=device, weights_only=False)

    from reamember.mops import memorize, remember

    eam = memorize(cfg, dataset=embeddings_dataset.test)
    memories_features, memories_recognition, memories_weights = remember(cfg, eam, dataset=embeddings_dataset.test)

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
    click.echo("[INFO] Classifying memories...")

    memories_recognition = []

    for f in tqdm(memories_features):
        f = torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
        memories_recognition.append(classifier.predict(f).cpu().numpy())

    memories_recognition = np.concatenate(memories_recognition, axis=0)
    original_labels = embeddings_dataset.test.targets.cpu().numpy()

    from sklearn.metrics import confusion_matrix
    from plotly import graph_objects as go

    cm = confusion_matrix(original_labels, memories_recognition)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=list(range(embeddings_dataset.n_classes)),
        y=list(range(embeddings_dataset.n_classes)),
        colorscale='Viridis',
        colorbar=dict(title='Count')
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        width=800,
        height=800
    )
    fig_path = path / "memories_confmatrix.html"
    click.echo(f"[INFO] Saving confusion matrix to: {fig_path}")
    fig.write_html(fig_path)

    #click.echo(f"[INFO] Saving memories to: {path / 'memories.pth'}")
    #torch.save({
    #    'features': memories_features,
    #    'recognition': memories_recognition,
    #    'weights': memories_weights
    #}, path / "memories.pth")

@cli.command()
@click.option('--config',  help='YAML configuration.')
def dream(config):
    "🛌 Not implemented"
    cfg = OmegaConf.load(config)
    click.echo(f"[INFO] Conf: {cfg}")
    path = f"experiments/{cfg.app.dataset}-{cfg.neural.latent_dim}"
    path = Path(path)
    pass

####################################################################################
# Utils
####################################################################################

@cli.command()
def launch_tensorboard():
    "Launch TensorBoard for monitoring."
    click.echo("[INFO] Running TensorBoard...")
    try:
        import subprocess
        subprocess.run(["tensorboard", f"--logdir={Path("logs")}", "--port=6006"], check=True)
    except FileNotFoundError:
        click.echo("[ERROR] TensorBoard not found. Please install it using 'pip install tensorboard'.")
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

if __name__ == "__main__":
    cli()