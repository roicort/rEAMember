import torch
import numpy as np
import pandas as pd
import json
from nltk.metrics import edit_distance
from tqdm import tqdm
from plotly import graph_objects as go
from reamember.eam.associative import NumpyAssociativeMemory as AssociativeMemory
from reamember.eam.mops import evalm_text_confusion, evalm_text, memorize
from reamember.utils import (
    ensure_directory,
    get_scalar_config_value,
    get_experiment_path,
    decode_text_embeddings,
    load_embeddings_dataset,
    Quant,
)
from omegaconf import ListConfig

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

def test_recall(cfg, device, experiments_root):

    latent = cfg.neural.latent_dim[0] if isinstance(cfg.neural.latent_dim, ListConfig) else int(cfg.neural.latent_dim)

    path = get_experiment_path(cfg, experiments_root, latent)
    embeddings_dataset = load_embeddings_dataset(path, device=device)

    # First Experiment: Test Recognition Rates

    recognition_text_path = ensure_directory(path / "recognition")

    quantizer = Quant(
        torch.cat([embeddings_dataset.train.data, embeddings_dataset.test.data], dim=0)
    ) # Create quantizer with train + test

    domains = (
        [int(domain) for domain in cfg.memory.domain]
        if isinstance(cfg.memory.domain, ListConfig)
        else [int(cfg.memory.domain)]
    )
    iotas = (
        [float(iota) for iota in cfg.memory.iota]
        if isinstance(cfg.memory.iota, ListConfig)
        else [float(cfg.memory.iota)]
    )
    kappas = (
        [float(kappa) for kappa in cfg.memory.kappa]
        if isinstance(cfg.memory.kappa, ListConfig)
        else [float(cfg.memory.kappa)]
    )
    xis = (
        [float(xi) for xi in cfg.memory.xi]
        if isinstance(cfg.memory.xi, ListConfig)
        else [float(cfg.memory.xi)]
    )
    sigmas = (
        [float(sigma) for sigma in cfg.memory.sigma]
        if isinstance(cfg.memory.sigma, ListConfig)
        else [float(cfg.memory.sigma)]
    )

    print(f"[INFO] Testing recall with latent={latent}, domains={domains}, sigmas={sigmas}, xis={xis}, iotas={iotas}, kappas={kappas}")

    train_embeddings = embeddings_dataset.train.data
    split_index = max(1, len(train_embeddings) // 2) # Split train into two halves for seen/unseen evaluation, ensuring at least one sample in the seen half
    memory_wrapper = EmbeddingDatasetWrapper(
        train=train_embeddings[:split_index],
        test=train_embeddings[split_index:],
    ) # Create a wrapper to hold the split train embeddings for memory filling and evaluation

    confusion_summaries = [] # Prepare to collect confusion summaries for all domains
    confusion_summary_path = (
        recognition_text_path / "recognition_confusion_summary.json"
    )

    from rich.progress import (
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
    ) as grid_progress:
        domain_task = grid_progress.add_task(
            f"[cyan]Domain: {domains[0]}",
            total=len(domains),
        )
        sigma_task = grid_progress.add_task(
            f"[magenta]Sigma: {sigmas[0]}",
            total=len(sigmas),
        )
        xi_task = grid_progress.add_task(
            f"[yellow]Xi: {xis[0]}",
            total=len(xis),
        )
        iota_task = grid_progress.add_task(
            f"[green]Iota: {iotas[0]}",
            total=len(iotas),
        )
        kappa_task = grid_progress.add_task(
            f"[red]Kappa: {kappas[0]}",
            total=len(kappas),
        )

        for domain in domains:
            grid_progress.reset(sigma_task)
            for sigma in sigmas:
                grid_progress.reset(xi_task)
                for xi in xis:
                    grid_progress.reset(iota_task)
                    for iota in iotas:
                        grid_progress.reset(kappa_task)
                        for kappa in kappas:

                            eam = AssociativeMemory(
                                n=latent,
                                m=domain,
                                xi=xi,
                                sigma=sigma,
                                iota=iota,
                                kappa=kappa,
                            )

                            # Fill with the first half of the training dataset
                            eam = memorize(
                                eam,
                                dataset=memory_wrapper.train,
                                quantizer=quantizer,
                                batch_size=cfg.memory.batch_size,
                            )

                            # Evaluate recognition on the seen and unseen halves
                            confusion = evalm_text_confusion(
                                eam,
                                seen_dataset=memory_wrapper.train,
                                unseen_dataset=memory_wrapper.test,
                                test_dataset=embeddings_dataset.test,
                                quantizer=quantizer,
                                batch_size=cfg.memory.batch_size,
                            )

                            confusion_payload = {
                                "dataset": cfg.app.dataset,
                                "latent_dim": int(latent),
                                "memory_domain": domain,
                                "sigma": sigma,
                                "iota": iota,
                                "kappa": kappa,
                                "xi": xi,
                                "seen_source": "first_half_of_train",
                                "unseen_source": "second_half_of_train",
                                "test_source": "dataset_test_split",
                                "labels": confusion["labels"],
                                "matrix": confusion["matrix"].tolist(),
                                "counts": confusion["counts"],
                                "rates": confusion["rates"],
                            }

                            confusion_path = (
                                recognition_text_path / f"recognition_confusion_domain_{domain}_sigma_{sigma}_xi_{xi}_iota_{iota}_kappa_{kappa}.json"
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
                                title=f"Text Memory Recognition Confusion Matrix (m={domain}) with σ={sigma}, iota={iota}, kappa={kappa}, xi={xi}",
                                xaxis_title="Memory Decision",
                                yaxis_title="Sample Type",
                                width=1200,
                                height=800,
                            )
                            fig.write_html(
                                recognition_text_path
                                / f"recognition_confusion_domain_{domain}_sigma_{sigma}_xi_{xi}_iota_{iota}_kappa_{kappa}.html"
                            )
                            fig.write_image(
                                recognition_text_path
                                / f"recognition_confusion_domain_{domain}_sigma_{sigma}_xi_{xi}_iota_{iota}_kappa_{kappa}.png"
                            )
                            click.echo(
                                f"[INFO] Recognition rates (m={domain}) | seen recognized: {confusion['rates']['seen_recognized_rate']:.4f} "
                                f"| unseen recognized: {confusion['rates']['unseen_recognized_rate']:.4f} "
                                f"| test recognized: {confusion['rates']['test_recognized_rate']:.4f}"
                            )
                            click.echo(f"[INFO] Recognition confusion saved to: {confusion_path}")
                            click.echo(
                                f"[INFO] Recognition confusion plot saved to: {recognition_text_path / f'recognition_confusion_domain_{domain}_sigma_{sigma}_xi_{xi}_iota_{iota}_kappa_{kappa}.html'}"
                            )

                            confusion_summaries.append(confusion_payload)
                            with open(confusion_summary_path, "w", encoding="utf-8") as f_out:
                                json.dump(
                                    confusion_summaries,
                                    f_out,
                                    indent=4,
                                    ensure_ascii=False,
                                )
                                grid_progress.update(
                                    kappa_task,
                                    advance=1,
                                    description=f"[red]Kappa: {kappa}",
                                )
                                grid_progress.update(
                                    iota_task,
                                    advance=1,
                                    description=f"[green]Iota: {iota}",
                                )
                            grid_progress.update(
                                xi_task,
                                advance=1,
                                description=f"[yellow]Xi: {xi}",
                            )
                        grid_progress.update(
                            sigma_task,
                            advance=1,
                            description=f"[magenta]Sigma: {sigma}",
                        )
                    grid_progress.update(
                        domain_task,
                        advance=1,
                        description=f"[cyan]Domains: {domain}",
                    )


def test_text_encoder(cfg, n_examples, device, experiments_root):

    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = TextDatasetWrapper(
        dataset_name=cfg.app.dataset,
        column=cfg.app.column,
    )

    latent = cfg.neural.latent_dim[0] if isinstance(cfg.neural.latent_dim, ListConfig) else int(cfg.neural.latent_dim)

    path = get_experiment_path(cfg, experiments_root, latent)
    transformer = create_sonar_model(device)
    embeddings_dataset = load_embeddings_dataset(path, device=device)
    transformer = create_sonar_model(device)
    reconstructed_text_path = ensure_directory(path / "reconstructed")
    total_test = len(embeddings_dataset.test.data)
    original_test_texts = [str(text) for text in dataset.test.texts[:total_test]]
    test_embeddings = embeddings_dataset.test.data[:total_test]
    
    # Seconda Experiment: Test embeddings without memory recall to evaluate the quality of the autoencoder independently
    samples, summary = text_reconstruction_metrics(
        model=transformer,
        device=device,
        original_texts=original_test_texts,
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

def get_text_embeddings(cfg, device, experiments_root):
    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = TextDatasetWrapper(
        dataset_name=cfg.app.dataset,
        column=cfg.app.column,
    )

    latent = cfg.neural.latent_dim[0] if isinstance(cfg.neural.latent_dim, ListConfig) else int(cfg.neural.latent_dim)
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
    sigma = float(get_scalar_config_value(cfg.memory.sigma))
    iota = float(get_scalar_config_value(cfg.memory.iota))
    kappa = float(get_scalar_config_value(cfg.memory.kappa))
    xi = float(get_scalar_config_value(cfg.memory.xi))

    click.echo(f"[INFO] Creating text memories with latent={latent}, domain={domain}")

    path = get_experiment_path(cfg, experiments_root, latent)
    embeddings_dataset = load_embeddings_dataset(path, device=device)
    dataset = TextDatasetWrapper(
        dataset_name=cfg.app.dataset,
        column=cfg.app.column,
    )

    all = torch.cat([embeddings_dataset.train.data, embeddings_dataset.test.data], dim=0)

    quantizer = Quant(all)

    eam = AssociativeMemory(
        n=latent,
        m=domain,
        xi=xi,
        sigma=sigma,
        iota=iota,
        kappa=kappa,
   )
    eam = memorize(
        eam,
        dataset=all,
        quantizer=quantizer,
        filling_percent=filling_percent,
        batch_size=cfg.memory.batch_size,
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
        quantizer=quantizer,
        batch_size=cfg.memory.batch_size,
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
        "sigma": sigma,
        "iota": iota,
        "kappa": kappa,
        "xi": xi,
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
    from omegaconf import ListConfig

    global_results = []

    msizes = cfg.memory.domain
    filling_percents = cfg.memory.filling
    iota_values = (
        [float(iota) for iota in cfg.memory.iota]
        if isinstance(cfg.memory.iota, ListConfig)
        else [float(cfg.memory.iota)]
    )
    kappa_values = (
        [float(kappa) for kappa in cfg.memory.kappa]
        if isinstance(cfg.memory.kappa, ListConfig)
        else [float(cfg.memory.kappa)]
    )
    xi_values = (
        [float(xi) for xi in cfg.memory.xi]
        if isinstance(cfg.memory.xi, ListConfig)
        else [float(cfg.memory.xi)]
    )
    sigma_values = (
        [float(sigma) for sigma in cfg.memory.sigma]
        if isinstance(cfg.memory.sigma, ListConfig)
        else [float(cfg.memory.sigma)]
    )

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
    sigma_task = progress.add_task(
        f"[magenta]Sigma: {sigma_values[0]}",
        total=len(sigma_values),
    )
    xi_task = progress.add_task(
        f"[cyan]Xi: {xi_values[0]}",
        total=len(xi_values),
    )
    iota_task = progress.add_task(
        f"[yellow]Iota: {iota_values[0]}",
        total=len(iota_values),
    )
    kappa_task = progress.add_task(
        f"[red]Kappa: {kappa_values[0]}",
        total=len(kappa_values),
    )

    progress.start()
    progress.start_task(latent_task)

    for latent in cfg.neural.latent_dim:
        path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
        embeddings_dataset = load_embeddings_dataset(path, device)

        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
            column=cfg.app.column,
        )

        transformer = create_sonar_model(device)

        train_embeddings = embeddings_dataset.train.data
        test_embeddings = embeddings_dataset.test.data
        all_embeddings = torch.cat([train_embeddings, test_embeddings], dim=0)
        quantizer = Quant(all_embeddings)
        test_texts = [str(text) for text in dataset.test.texts]

        results = []

        progress.start_task(msize_task)
        progress.start_task(sigma_task)
        progress.start_task(xi_task)
        progress.start_task(iota_task)
        progress.start_task(kappa_task)

        for msize in msizes:
            for filling_percent in filling_percents:
                progress.reset(sigma_task)
                for sigma in sigma_values:
                    progress.reset(xi_task)
                    for xi in xi_values:
                        progress.reset(iota_task)
                        for iota in iota_values:
                            progress.reset(kappa_task)
                            for kappa in kappa_values:
                                eam = AssociativeMemory(
                                    n=latent,
                                    m=msize,
                                    sigma=sigma,
                                    xi=xi,
                                    iota=iota,
                                    kappa=kappa,
                                )

                                eam = memorize(
                                    eam,
                                    dataset=all_embeddings,
                                    quantizer=quantizer,
                                    filling_percent=filling_percent,
                                    batch_size=cfg.memory.batch_size,
                                )

                                (
                                    memories_features,
                                    recognitions,
                                    _weights,
                                    recognized,
                                    unrecognized,
                                ) = evalm_text(
                                    eam,
                                    dataset=embeddings_dataset.test,
                                    quantizer=quantizer,
                                    batch_size=cfg.memory.batch_size,
                                )

                                recognized_embeddings = memories_features[recognitions]
                                recognized_texts = [
                                    text
                                    for text, is_recognized in zip(test_texts, recognitions)
                                    if is_recognized
                                ]

                                _, summary = text_reconstruction_metrics(
                                    model=transformer,
                                    device=device,
                                    original_texts=recognized_texts,
                                    embeddings=recognized_embeddings,
                                )

                                results.append(
                                    {
                                        "latent": latent,
                                        "msize": msize,
                                        "filling_percent": filling_percent,
                                        "sigma": sigma,
                                        "xi": xi,
                                        "iota": iota,
                                        "kappa": kappa,
                                        "recognized": recognized,
                                        "unrecognized": unrecognized,
                                        "mean_cosine": summary["mean_cosine"],
                                        "mean_l2": summary["mean_l2"],
                                        "mean_edit_distance": summary["mean_edit_distance"],
                                    }
                                )
                                progress.update(
                                    kappa_task,
                                    advance=1,
                                    description=f"[red]Kappa: {kappa}",
                                )
                            progress.update(
                                iota_task,
                                advance=1,
                                description=f"[yellow]Iota: {iota}",
                            )
                        progress.update(
                            xi_task,
                            advance=1,
                            description=f"[cyan]Xi: {xi}",
                        )
                    progress.update(
                        sigma_task,
                        advance=1,
                        description=f"[magenta]Sigma: {sigma}",
                    )

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

        path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
        save_path = path / f"{latent}_parameters_search_results.json"
        click.echo(f"[INFO] Saving results to: {save_path}")
        with open(save_path, "w") as f:
            json.dump(global_results, f, indent=4)

    progress.stop()

    path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
    save_path = path / "global_memories_results.json"
    click.echo(f"[INFO] Saving results to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(global_results, f, indent=4)

    df = pd.DataFrame(global_results)
    df = df.sort_values(
        by=["mean_cosine", "recognized", "mean_edit_distance", "mean_l2"],
        ascending=[False, False, True, True],
    )

    best_params = df.iloc[0].to_dict()
    cfg.neural.latent_dim = [int(best_params["latent"])]
    cfg.memory.domain = [int(best_params["msize"])]
    cfg.memory.filling = [float(best_params["filling_percent"])]
    cfg.memory.sigma = float(best_params["sigma"])
    cfg.memory.xi = float(best_params["xi"])
    cfg.memory.iota = float(best_params["iota"])
    cfg.memory.kappa = float(best_params["kappa"])

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