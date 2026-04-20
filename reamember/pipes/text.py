import torch
import numpy as np
import pandas as pd
import json
from nltk.metrics import edit_distance
from tqdm import tqdm
from plotly import graph_objects as go
from reamember.eam.mops import evalm_text_confusion, evalm_text, memorize
from reamember.utils import (
    ensure_directory,
    get_scalar_config_value,
    get_experiment_path,
    decode_text_embeddings,
    load_embeddings_dataset,
    create_associative_memory,
    _get_quantization_bounds,
)

from reamember.neuralnets.transformer import SONAR
from reamember.embeddings import get_embeddings as build_embeddings
import rich_click as click
from reamember.datasets.text import TextDatasetWrapper
from reamember.datasets.embedding import EmbeddingDatasetWrapper
from omegaconf import OmegaConf
from pathlib import Path
import re

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


def test_text_encoder(cfg, n_examples, device, experiments_root):
    from omegaconf import ListConfig

    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = TextDatasetWrapper(
        dataset_name=cfg.app.dataset,
        column=cfg.app.column,
    )

    for latent in cfg.neural.latent_dim:
        path = get_experiment_path(cfg, experiments_root, latent)

        transformer = create_sonar_model(device)
        embeddings_dataset = load_embeddings_dataset(path, device=device)
        reconstructed_text_path = ensure_directory(path / "reconstructed")
        global_quantize_min, global_quantize_max = _get_quantization_bounds(
            embeddings_dataset.train.data,
            embeddings_dataset.test.data,
        )

        total = len(embeddings_dataset.test.data)

        original_texts = [str(text) for text in dataset.test.texts[:total]]
        test_embeddings = embeddings_dataset.test.data[:total]

        # Recognition evaluation across memory domains

        recognition_text_path = ensure_directory(path / "recognition")

        domains = (
            [int(domain) for domain in cfg.memory.domain]
            if isinstance(cfg.memory.domain, ListConfig)
            else [int(cfg.memory.domain)]
        )

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

                # Fill with the first half of the training dataset
                eam = memorize(
                    eam,
                    dataset=memory_wrapper.train,
                    quantize_min=global_quantize_min,
                    quantize_max=global_quantize_max,
                )

                # Evaluate recognition on the seen and unseen halves
                confusion = evalm_text_confusion(
                    eam,
                    seen_dataset=memory_wrapper.train,
                    unseen_dataset=memory_wrapper.test,
                    test_dataset=embeddings_dataset.test,
                    quantize_min=global_quantize_min,
                    quantize_max=global_quantize_max,
                )

                confusion_payload = {
                    "dataset": cfg.app.dataset,
                    "latent_dim": int(latent),
                    "memory_domain": domain,
                    "sigma": cfg.memory.sigma,
                    "seen_source": "first_half_of_train",
                    "unseen_source": "second_half_of_train",
                    "test_source": "dataset_test_split",
                    "labels": confusion["labels"],
                    "matrix": confusion["matrix"].tolist(),
                    "counts": confusion["counts"],
                    "rates": confusion["rates"],
                }
                confusion_summaries.append(confusion_payload)

                confusion_path = (
                    recognition_text_path / f"recognition_confusion_domain_{domain}.json"
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
                    title=f"Text Memory Recognition Confusion Matrix (m={domain}) with σ={cfg.memory.sigma}",
                    xaxis_title="Memory Decision",
                    yaxis_title="Sample Type",
                    width=1200,
                    height=800,
                )
                fig.write_html(
                    recognition_text_path
                    / f"recognition_confusion_domain_{domain}.html"
                )
                fig.write_image(
                    recognition_text_path
                    / f"recognition_confusion_domain_{domain}.png"
                )

                click.echo(
                    f"[INFO] Recognition rates (m={domain}) | seen recognized: {confusion['rates']['seen_recognized_rate']:.4f} "
                    f"| unseen recognized: {confusion['rates']['unseen_recognized_rate']:.4f} "
                    f"| test recognized: {confusion['rates']['test_recognized_rate']:.4f}"
                )
                click.echo(f"[INFO] Recognition confusion saved to: {confusion_path}")
                click.echo(
                    f"[INFO] Recognition confusion plot saved to: {recognition_text_path / f'recognition_confusion_domain_{domain}.html'}"
                )
                domain_progress.update(domain_task, advance=1)

        confusion_summary_path = (
            recognition_text_path / "recognition_confusion_summary.json"
        )
        with open(confusion_summary_path, "w", encoding="utf-8") as f_out:
            json.dump(
                confusion_summaries,
                f_out,
                indent=4,
                ensure_ascii=False,
            )

        sorted_confusions = sorted(
            confusion_summaries,
            key=lambda item: item["memory_domain"],
        )
        domain_values = [item["memory_domain"] for item in sorted_confusions]
        seen_recognized_rates = [
            item["rates"]["seen_recognized_rate"] for item in sorted_confusions
        ]
        unseen_recognized_rates = [
            item["rates"]["unseen_recognized_rate"] for item in sorted_confusions
        ]
        test_recognized_rates = [
            item["rates"]["test_recognized_rate"] for item in sorted_confusions
        ]
        seen_unrecognized_rates = [
            item["rates"]["seen_unrecognized_rate"] for item in sorted_confusions
        ]
        unseen_unrecognized_rates = [
            item["rates"]["unseen_unrecognized_rate"] for item in sorted_confusions
        ]
        test_unrecognized_rates = [
            item["rates"]["test_unrecognized_rate"] for item in sorted_confusions
        ]

        rates_fig = go.Figure()
        rates_fig.add_trace(
            go.Scatter(
                x=domain_values,
                y=seen_recognized_rates,
                mode="lines+markers",
                name="Seen recognized",
            )
        )
        rates_fig.add_trace(
            go.Scatter(
                x=domain_values,
                y=unseen_recognized_rates,
                mode="lines+markers",
                name="Unseen recognized",
            )
        )
        rates_fig.add_trace(
            go.Scatter(
                x=domain_values,
                y=test_recognized_rates,
                mode="lines+markers",
                name="Test recognized",
            )
        )
        rates_fig.add_trace(
            go.Scatter(
                x=domain_values,
                y=seen_unrecognized_rates,
                mode="lines+markers",
                name="Seen unrecognized",
                line=dict(dash="dash"),
            )
        )
        rates_fig.add_trace(
            go.Scatter(
                x=domain_values,
                y=unseen_unrecognized_rates,
                mode="lines+markers",
                name="Unseen unrecognized",
                line=dict(dash="dash"),
            )
        )
        rates_fig.add_trace(
            go.Scatter(
                x=domain_values,
                y=test_unrecognized_rates,
                mode="lines+markers",
                name="Test unrecognized",
                line=dict(dash="dash"),
            )
        )
        rates_fig.update_layout(
            title=(
                "Text Memory Recognition Rates by Domain "
                f"(latent={latent}, σ={cfg.memory.sigma})"
            ),
            xaxis_title="Memory domain (m)",
            yaxis_title="Rate",
            yaxis=dict(range=[0.0, 1.0]),
            width=1400,
            height=800,
            legend_title="Metric",
        )

        rates_plot_path = recognition_text_path / "recognition_rates_by_domain.html"
        rates_image_path = recognition_text_path / "recognition_rates_by_domain.png"
        rates_fig.write_html(rates_plot_path)
        rates_fig.write_image(rates_image_path)

        # Test embeddings without memory recall to evaluate the quality of the autoencoder independently
        samples, summary = text_reconstruction_metrics(
            model=transformer,
            device=device,
            original_texts=original_texts,
            embeddings=test_embeddings,
            batch_size=64,
        )

        reconstructed_samples_path = reconstructed_text_path / "reconstructed.json"
        with open(reconstructed_samples_path, "w", encoding="utf-8") as f_out:
            json.dump(samples, f_out, indent=4, ensure_ascii=False)

        metrics_path = reconstructed_text_path / "metrics.json"
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

        click.echo(
            f"[INFO] Reconstruction metrics | cosine: {summary['mean_cosine']:.4f} "
            f"| l2: {summary['mean_l2']:.4f} "
            f"| edit distance: {summary['mean_edit_distance']:.4f}"
        )

        click.echo(f"[INFO] Reconstructed texts saved to: {reconstructed_samples_path}")
        click.echo(f"[INFO] Reconstruction metrics saved to: {metrics_path}")
        click.echo(
            f"[INFO] Recognition confusion summary saved to: {confusion_summary_path}"
        )
        click.echo(f"[INFO] Recognition rates plot saved to: {rates_plot_path}")


def get_text_embeddings(cfg, device, experiments_root):
    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = TextDatasetWrapper(
        dataset_name=cfg.app.dataset,
        column=cfg.app.column,
    )

    for latent in cfg.neural.latent_dim:
        path = get_experiment_path(cfg, experiments_root, latent)
        model = create_sonar_model(device)

        build_embeddings(
            model,
            dataset,
            modality=cfg.app.modality,
            device=device,
            save_path=path,
        )


def create_text_memories(cfg, n_saved, device, experiments_root):
    latent = int(get_scalar_config_value(cfg.neural.latent_dim))
    domain = int(get_scalar_config_value(cfg.memory.domain))
    filling_percent = float(get_scalar_config_value(cfg.memory.filling))

    click.echo(f"[INFO] Creating text memories with latent={latent}, domain={domain}")

    path = get_experiment_path(cfg, experiments_root, latent)
    embeddings_dataset = load_embeddings_dataset(path, device=device)
    dataset = TextDatasetWrapper(
        dataset_name=cfg.app.dataset,
        column=cfg.app.column,
    )

    all = torch.cat([embeddings_dataset.train.data, embeddings_dataset.test.data], dim=0)

    global_quantize_min, global_quantize_max = _get_quantization_bounds(
        all
    )

    eam = create_associative_memory(cfg, latent, domain)
    eam = memorize(
        eam,
        dataset=all,
        quantize_max=global_quantize_max,
        quantize_min=global_quantize_min,
        filling_percent=filling_percent,
    )
    
    (
        memories_features,
        recognitions,
        weights,
        recognized_rate,
        unrecognized_rate,
    ) = evalm_text(
        eam,
        dataset=embeddings_dataset.test,
        quantize_min=global_quantize_min,
        quantize_max=global_quantize_max,
    )

    total = len(memories_features)

    transformer = create_sonar_model(device)
    output_path = ensure_directory(path / f"dim{domain}_memory_reconstructed")

    all_texts = [str(text) for text in dataset.test.texts[:total]]
    recognized_indices = np.flatnonzero(recognitions)
    reconstructable_texts = [all_texts[index] for index in recognized_indices]
    reconstructable_embeddings = memories_features[recognitions]

    samples, summary = text_reconstruction_metrics(
        model=transformer,
        device=device,
        original_texts=reconstructable_texts,
        embeddings=reconstructable_embeddings,
    )

    for sample, original_index in zip(samples, recognized_indices):
        sample["index"] = int(original_index)
        sample["recognized"] = True
        sample["weight"] = (
            float(weights[original_index])
            if np.isscalar(weights[original_index])
            else float(np.asarray(weights[original_index]).mean())
        )

    saved_samples = samples if n_saved == 0 else samples[:n_saved]

    reconstructed_samples_path = output_path / "reconstructed.json"
    with open(reconstructed_samples_path, "w", encoding="utf-8") as f_out:
        json.dump(saved_samples, f_out, indent=4, ensure_ascii=False)

    payload = {
        "dataset": cfg.app.dataset,
        "column": cfg.app.column,
        "latent_dim": latent,
        "memory_domain": domain,
        "filling_percent": filling_percent,
        "sigma": cfg.memory.sigma,
        "recognition": {
            "recognized_rate": float(recognized_rate),
            "unrecognized_rate": float(unrecognized_rate),
            "recognized_count": int(np.sum(recognitions)),
            "unrecognized_count": int(total - np.sum(recognitions)),
            "samples": int(total),
        },
        "summary": summary,
    }

    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, indent=4, ensure_ascii=False)

    click.echo(
        f"[INFO] Text memory recognition | recognized: {recognized_rate:.4f} | unrecognized: {unrecognized_rate:.4f}"
    )
    click.echo(
        f"[INFO] Text memory reconstruction | cosine: {summary['mean_cosine']:.4f} | l2: {summary['mean_l2']:.4f} | edit distance: {summary['mean_edit_distance']:.4f}"
    )
    click.echo(f"[INFO] Reconstructed texts saved to: {reconstructed_samples_path}")
    click.echo(f"[INFO] Text memory metrics saved to: {metrics_path}")

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
    source_embeddings = source_embeddings.detach().cpu()

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


def get_besttext_params(cfg, config, device, EXPERIMENTS_ROOT):
    
    from sklearn.model_selection import KFold

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
        path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
        embeddings_dataset = load_embeddings_dataset(path, device)

        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
            column=cfg.app.column,
        )

        quantize_min = torch.min(
            torch.cat([embeddings_dataset.train.data, embeddings_dataset.test.data], dim=0)
        )
        quantize_max = torch.max(
            torch.cat([embeddings_dataset.train.data, embeddings_dataset.test.data], dim=0)
        )

        transformer = create_sonar_model(device)

        X = embeddings_dataset.train.data
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        texts = np.array([str(text) for text in dataset.train.texts], dtype=object)

        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

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
                    quantize_min, quantize_max = _get_quantization_bounds(X_train, X_val)

                    fold_train_wrapper = EmbeddingDatasetWrapper(
                        train=torch.tensor(X_train, dtype=torch.float32),
                        test=torch.tensor(X_val, dtype=torch.float32),
                    )

                    eam = create_associative_memory(cfg, latent, msize)

                    eam = memorize(
                        eam,
                        dataset=fold_train_wrapper.train,
                        quantize_min=quantize_min,
                        quantize_max=quantize_max,
                        filling_percent=filling_percent,
                    )

                    (
                        memories_features,
                        recognitions,
                        _weights,
                        recognized,
                        unrecognized,
                    ) = evalm_text(
                        eam,
                        dataset=fold_train_wrapper.test,
                        quantize_min=quantize_min,
                        quantize_max=quantize_max,
                    )

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