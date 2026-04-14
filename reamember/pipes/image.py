import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import rich_click as click
import torch
import torchvision

from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from omegaconf import OmegaConf

from reamember.datasets.embedding import EmbeddingDatasetWrapper
from reamember.datasets.image import ImageDatasetWrapper
from reamember.eam.associative import NumpyAssociativeMemory as AssociativeMemory
from reamember.eam.mops import evalm, memorize, remember
from reamember.neuralnets.autoencoder import Autoencoder
from reamember.neuralnets.classifier import Classifier
from reamember.utils import (
	ensure_directory,
	create_associative_memory,
	get_scalar_config_value,
	get_experiment_path,
	load_embeddings_dataset,
	load_model_state,
)


def get_bestimage_params(
	cfg,
	config,
	device,
	experiments_root,
	task_name="GetBestParams",
):
	from sklearn.model_selection import StratifiedKFold

	global_results = []

	msizes = cfg.memory.domain
	filling_percents = cfg.memory.filling
	folds = cfg.app.crossval.folds
	seed = cfg.app.crossval.seed

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
	name_task = progress.add_task(f"[bold white]{task_name}[/bold white]", total=None)

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
		path = get_experiment_path(cfg, experiments_root, latent)
		embeddings_dataset = load_embeddings_dataset(path, device=device)

		dataset = ImageDatasetWrapper(
			dataset_name=cfg.app.dataset,
		)

		input_shape = dataset.train[0][0].shape
		click.echo(f"[INFO] Input shape: {input_shape}")

		x_values = embeddings_dataset.train.data
		y_values = embeddings_dataset.train.targets
		if isinstance(y_values, torch.Tensor):
			y_values = y_values.cpu().numpy()
		if isinstance(x_values, torch.Tensor):
			x_values = x_values.cpu().numpy()
		skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

		classifier = Classifier(
			latent_dim=latent,
			n_classes=embeddings_dataset.n_classes,
		)

		classifier_path = path / "classifier.pth"
		load_model_state(classifier, classifier_path, "Classifier", device=device)

		decoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
		decoder_path = path / "autoencoder.pth"
		load_model_state(decoder, decoder_path, "Decoder", device=device)

		results = []

		progress.start_task(msize_task)
		progress.start_task(filling_task)
		progress.start_task(fold_task)

		for msize in msizes:
			for filling_percent in filling_percents:
				fold_metrics = []
				progress.reset(fold_task)
				for train_idx, val_idx in skf.split(x_values, y_values):
					x_train, y_train = x_values[train_idx], y_values[train_idx]
					x_val, y_val = x_values[val_idx], y_values[val_idx]

					fold_train_wrapper = EmbeddingDatasetWrapper(
						train=torch.tensor(x_train),
						test=torch.tensor(x_val),
						labels_train=torch.tensor(y_train),
						labels_test=torch.tensor(y_val),
					)

					eam = create_associative_memory(cfg, latent, msize)

					eam, min_value, max_value = memorize(
						eam,
						dataset=fold_train_wrapper.train,
						filling_percent=filling_percent,
					)

					percentages, recall, precision = evalm(
						eam,
						classifier=classifier,
						dataset=fold_train_wrapper.test,
						min_value=min_value,
						max_value=max_value,
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
				msize_task, advance=1, description=f"[green]Memory Size: {msize}"
			)

		progress.reset(msize_task)
		global_results.extend(results)
		progress.update(
			latent_task,
			advance=1,
			description=f"[magenta]Latent: {latent}",
		)

	progress.stop_task(name_task)
	progress.update(name_task, description=f"[bold green]{task_name}[/bold green]")
	progress.stop()

	path = get_experiment_path(cfg, experiments_root, latent)
	save_path = path / "memories_results.json"
	click.echo(f"[INFO] Saving results to: {save_path}")
	with open(save_path, "w") as f:
		json.dump(global_results, f, indent=4)

	df = pd.DataFrame(global_results)
	df = df.sort_values(by=["recognized", "precision"], ascending=False)

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

	click.echo("[INFO] Best parameters search completed.")


def test_image_encoder(cfg, n, device, experiments_root):
	click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

	dataset = ImageDatasetWrapper(
		dataset_name=cfg.app.dataset,
	)

	input_shape = dataset.test[0][0].shape

	for latent in cfg.neural.latent_dim:
		path = get_experiment_path(cfg, experiments_root, latent)

		autoencoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
		encoder_path = path / "autoencoder.pth"

		load_model_state(autoencoder, encoder_path, "Encoder", device=device)
		embeddings_dataset = load_embeddings_dataset(path, device)
		reconstructed_img_path = ensure_directory(path / "reconstructed")

		for index in tqdm(
			range(len(embeddings_dataset.test.data))
			if n == 0
			else range(min(n, len(embeddings_dataset.test.data)))
		):
			feature = torch.as_tensor(
				embeddings_dataset.test.data[index],
				dtype=torch.float32,
				device=device,
			).unsqueeze(0)
			with torch.no_grad():
				reconstructed = autoencoder.decode(feature)
				torchvision.utils.save_image(
					reconstructed, reconstructed_img_path / f"img_{index}.png"
				)

		click.echo(
			f"[INFO] Reconstructed images saved to: {reconstructed_img_path}"
		)


def create_image_memories(cfg, n, device, experiments_root):
	latent = int(get_scalar_config_value(cfg.neural.latent_dim))
	domain = int(get_scalar_config_value(cfg.memory.domain))

	click.echo(f"[INFO] Creating memories with latent={latent}, domain={domain}")

	path = get_experiment_path(cfg, experiments_root, latent)
	embeddings_dataset = torch.load(
		path / "embeddings.pth", map_location=device, weights_only=False
	)

	dataset = ImageDatasetWrapper(
		dataset_name=cfg.app.dataset,
	)

	input_shape = dataset.train[0][0].shape
	click.echo(f"[INFO] Input shape: {input_shape}")

	eam = create_associative_memory(cfg, latent, domain)
	eam, min_value, max_value = memorize(eam, dataset=embeddings_dataset.train)

	memories_features, memories_recognition, _ = remember(
		cfg,
		eam=eam,
		dataset=embeddings_dataset.test,
		min_value=min_value,
		max_value=max_value,
	)

	click.echo("[INFO] Classifying memories...")

	classifier = Classifier(
		latent_dim=latent,
		n_classes=embeddings_dataset.n_classes,
	)
	classifier_path = path / "classifier.pth"
	load_model_state(classifier, classifier_path, "Classifier", device=device)

	decoder = Autoencoder(input_shape=input_shape, latent_dim=latent)
	decoder_path = path / "autoencoder.pth"
	load_model_state(decoder, decoder_path, "Decoder", device=device)

	reconstructed_img_path = ensure_directory(path / f"dim{domain}_memory_reconstructed")
	recognition_predictions = []

	for index in tqdm(range(len(memories_features))):
		feature = torch.as_tensor(
			memories_features[index], dtype=torch.float32, device=device
		).unsqueeze(0)
		with torch.no_grad():
			recognition_predictions.append(classifier.predict(feature).cpu().numpy())

	for index in tqdm(
		range(len(memories_features))
		if n == 0
		else range(min(n, len(memories_features)))
	):
		feature = torch.as_tensor(
			memories_features[index], dtype=torch.float32, device=device
		).unsqueeze(0)
		torchvision.utils.save_image(
			decoder.decode(feature).cpu(), reconstructed_img_path / f"img_{index}.png"
		)

	recognition_predictions = np.concatenate(recognition_predictions, axis=0)
	original_labels = embeddings_dataset.test.targets.cpu().numpy()

	cm = confusion_matrix(original_labels, recognition_predictions)
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


def run_image_dream(cfg, num_cycles, init_type, idx, device, experiments_root):
	latent = int(get_scalar_config_value(cfg.neural.latent_dim))
	path = get_experiment_path(cfg, experiments_root, latent)

	embeddings_dataset = load_embeddings_dataset(path, device)
	embeddings = (
		embeddings_dataset.test.data
		if hasattr(embeddings_dataset, "test")
		else embeddings_dataset["test"]["data"]
	)

	mem_params = cfg.memory if hasattr(cfg, "memory") else {}
	n = latent
	m = cfg.memory.m if hasattr(cfg.memory, "m") else 4
	memory = AssociativeMemory(n=n, m=m, device=device, **mem_params)

	train_embeddings = (
		embeddings_dataset.train.data
		if hasattr(embeddings_dataset, "train")
		else embeddings_dataset["train"]["data"]
	)
	for vector in train_embeddings:
		memory.register(vector.to(device))

	input_shape = (
		embeddings_dataset.input_shape
		if hasattr(embeddings_dataset, "input_shape")
		else (1, 28, 28)
	)
	decoder = Autoencoder(input_shape=input_shape, latent_dim=n)
	decoder_path = path / "autoencoder.pth"
	load_model_state(decoder, decoder_path, "Decoder", device=device)
	decoder.eval()

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

	for index in range(num_cycles):
		recalled, accepted, weight = memory.recall(vector)
		decoded = decoder.decode(
			torch.tensor(recalled, dtype=torch.float32, device=device).unsqueeze(0)
		)
		torchvision.utils.save_image(decoded, dreams_path / f"dream_{index}.png")
		vector = torch.tensor(recalled, dtype=torch.float32, device=device)

	click.echo(f"[INFO] Sueños guardados en: {dreams_path}")

def plot_image_memory_results(cfg, experiments_root):
	latent = int(get_scalar_config_value(cfg.neural.latent_dim))
	filling_percent = float(get_scalar_config_value(cfg.memory.filling))
	path = get_experiment_path(cfg, experiments_root, latent)
	load_path = path / "memories_results.json"
	click.echo(f"[INFO] Loading results from: {load_path}")

	with open(load_path) as f:
		data = json.load(f)
		df = pd.DataFrame(data)
		click.echo(
			f"[INFO] Filtered results for filling percent '{filling_percent}':"
		)
		newdf = df[df["filling_percent"] == filling_percent]

		for latent in newdf["latent"].unique():
			subset = newdf[newdf["latent"] == latent]
			fig1 = px.bar(
				subset,
				x=subset["msize"].astype(str),
				y=["unrecognized", "correct", "incorrect"],
				color_discrete_map={
					"unrecognized": "blue",
					"correct": "green",
					"incorrect": "red",
				},
				title=f"Filling Percent: {filling_percent}, Latent: {latent}",
			)
			fig1.update_xaxes(type="category")

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
