import torch
import numpy as np
import pandas as pd
import json
import math
import string
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


def get_memory_batch_size(cfg, domain):
    batch_size = cfg.memory.batch_size
    if not isinstance(batch_size, (list, tuple, ListConfig)):
        return batch_size
    if len(batch_size) == 0:
        return None
    if len(batch_size) != len(cfg.memory.domain):
        return batch_size[0]

    domain_index = list(cfg.memory.domain).index(domain)
    return batch_size[domain_index]

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


def normalize_noise_level(noise_level):
    noise_level = float(noise_level)
    if noise_level < 0:
        return 0.0
    if noise_level > 1:
        noise_level /= 100.0
    return min(noise_level, 1.0)


def get_random_replacement_char(original_char):
    if str(original_char).isupper():
        alphabet = string.ascii_uppercase
        original = str(original_char).upper()
    else:
        alphabet = string.ascii_lowercase
        original = str(original_char).lower()

    candidates = [char for char in alphabet if char != original]
    return str(np.random.choice(candidates))


def apply_text_noise(text, noise_level=0.1, character_mask=None):
    text = str(text).strip()
    if not text:
        return text

    normalized_noise = normalize_noise_level(noise_level)
    if normalized_noise <= 0:
        return text

    replaceable_positions = [
        index for index, char in enumerate(text) if char.isalpha()
    ]
    if not replaceable_positions:
        replaceable_positions = [
            index for index, char in enumerate(text) if not char.isspace()
        ]
    if not replaceable_positions:
        return text

    n_positions = len(replaceable_positions)
    n_to_mask = min(
        n_positions,
        max(1, math.ceil(n_positions * normalized_noise)),
    )
    selected_positions = np.random.choice(
        replaceable_positions,
        size=n_to_mask,
        replace=False,
    )

    masked_text = list(text)
    for position in selected_positions:
        masked_text[position] = (
            get_random_replacement_char(masked_text[position])
            if character_mask is None
            else character_mask
        )

    return ''.join(masked_text)


def test_recall(cfg, device, experiments_root, use_noise=False):

    latent = cfg.neural.latent_dim[0] if isinstance(cfg.neural.latent_dim, ListConfig) else int(cfg.neural.latent_dim)

    path = get_experiment_path(cfg, experiments_root, latent)
    embeddings_dataset = load_embeddings_dataset(path, device=device)
    recognition_dirname = "recognition"

    if use_noise:
        embeddings_dataset_noise = load_embeddings_dataset(
            path,
            device=device,
            noised=True,
        )
        recognition_dirname = f"recognition_noise_{cfg.app.noise}"
        quant_source = torch.cat([embeddings_dataset.train.data, embeddings_dataset_noise.train.data, embeddings_dataset_noise.test.data], dim=0)
        test_embeddings_dataset = embeddings_dataset_noise

        memory_wrapper = EmbeddingDatasetWrapper(
            train=embeddings_dataset.train.data,
            test=embeddings_dataset_noise.train.data,
        ) # Fill memory with clean train and evaluate on noised train for seen/unseen evaluation, using noised test for final test evaluation
        
        seen_source = "clean_train"
        unseen_source = "noised_train"
        test_source = "noised_test"

    else:
        test_embeddings_dataset = embeddings_dataset
        quant_source = torch.cat([embeddings_dataset.train.data, embeddings_dataset.test.data], dim=0)
        train_embeddings = embeddings_dataset.train.data
        split_index = max(1, len(train_embeddings) // 2) # Split train into two halves for seen/unseen evaluation, ensuring at least one sample in the seen half
        memory_wrapper = EmbeddingDatasetWrapper(
            train=train_embeddings[:split_index],
            test=train_embeddings[split_index:],
        ) # Fill memory with first half of train and evaluate on second half of train for seen/unseen evaluation, using test for final test evaluation
        
        seen_source = "first_half_of_train"
        unseen_source = "second_half_of_train"
        test_source = "test"

    quantizer = Quant( quant_source ) # Create quantizer with train + test

    # First Experiment: Test Recognition Rates

    recognition_text_path = ensure_directory(path / recognition_dirname)

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

    print(
        f"[INFO] Testing recall with latent={latent}, domains={domains}, sigmas={sigmas}, "
        f"xis={xis}, iotas={iotas}, kappas={kappas}, use_noise={use_noise}"
    )

    confusion_summaries = [] # Prepare to collect confusion summaries for all domains
    confusion_summary_path = (
        recognition_text_path / f"recognition_confusion_summary_{use_noise}.json"
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
            batch_size = get_memory_batch_size(cfg, domain)
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

                            # Memorize using the specified memory parameters, quantizer built from train + test and the appropriate train split for seen/unseen evaluation
                            eam = memorize(
                                eam,
                                dataset=memory_wrapper.train,
                                quantizer=quantizer,
                                batch_size=batch_size,
                            )

                            # Evaluate recognition on the seen and unseen halves
                            confusion = evalm_text_confusion(
                                eam,
                                seen_dataset=memory_wrapper.train,
                                unseen_dataset=memory_wrapper.test,
                                test_dataset=test_embeddings_dataset.test,
                                quantizer=quantizer,
                                batch_size=batch_size,
                            )

                            confusion_payload = {
                                "dataset": cfg.app.dataset,
                                "latent_dim": int(latent),
                                "memory_domain": domain,
                                "sigma": sigma,
                                "iota": iota,
                                "kappa": kappa,
                                "xi": xi,
                                "seen_source": seen_source,
                                "unseen_source": unseen_source,
                                "test_source": test_source,
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
                                / f"recognition_confusion_domain_{domain}_sigma_{sigma}_xi_{xi}_iota_{iota}_kappa_{kappa}.svg"
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
        seed=cfg.app.seed,
    )

    latent = cfg.neural.latent_dim[0] if isinstance(cfg.neural.latent_dim, ListConfig) else int(cfg.neural.latent_dim)

    path = get_experiment_path(cfg, experiments_root, latent)
    transformer = create_sonar_model(device)
    embeddings_dataset = load_embeddings_dataset(path, device=device)
    reconstructed_text_path = ensure_directory(path / "reconstructed")
    total_test = len(embeddings_dataset.test.data)
    original_test_texts = [str(text) for text in dataset.test.texts[:total_test]]

    click.echo(
        f"[INFO] Evaluating reconstruction with latent={latent}, source=dataset_test_split"
    )
    clean_samples, clean_summary = text_reconstruction_metrics(
        model=transformer,
        device=device,
        original_texts=original_test_texts,
        embeddings=embeddings_dataset.test.data[:total_test],
        batch_size=64,
    )

    noise_samples = None
    noise_summary = None
    if cfg.app.noise is not None and float(cfg.app.noise) > 0:
        noised_embeddings_dataset = load_embeddings_dataset(
            path,
            device=device,
            noised=True,
        )
        noised_test_texts = np.load(
            path / "noised_test_texts.npy",
            allow_pickle=True,
        ).tolist()[:total_test]

        click.echo(
            f"[INFO] Evaluating reconstruction with latent={latent}, source=noised_dataset_test_split"
        )
        noise_samples, noise_summary = text_reconstruction_metrics(
            model=transformer,
            device=device,
            original_texts=noised_test_texts,
            embeddings=noised_embeddings_dataset.test.data[:total_test],
            batch_size=64,
        )

    aligned_samples = []
    for index, clean_sample in enumerate(clean_samples):
        sample = {
            "index": int(clean_sample["index"]),
            "original": clean_sample["original"],
            "clean": {
                "cue": clean_sample["cue"],
                "reconstructed": clean_sample["reconstructed"],
                "cosine": clean_sample["cosine"],
                "euclidean": clean_sample["euclidean"],
                "l2": clean_sample["l2"],
                "edit_distance": clean_sample["edit_distance"],
            },
        }

        if noise_samples is not None:
            noise_sample = noise_samples[index]
            sample["noise"] = {
                "cue": noise_sample["cue"],
                "reconstructed": noise_sample["reconstructed"],
                "cosine": noise_sample["cosine"],
                "euclidean": noise_sample["euclidean"],
                "l2": noise_sample["l2"],
                "edit_distance": noise_sample["edit_distance"],
            }

        aligned_samples.append(sample)

    saved_aligned_samples = (
        aligned_samples if n_examples == 0 else aligned_samples[:n_examples]
    )
    aligned_samples_path = reconstructed_text_path / "reconstructed.json"
    with open(aligned_samples_path, "w", encoding="utf-8") as f_out:
        json.dump(saved_aligned_samples, f_out, indent=4, ensure_ascii=False)

    aligned_metrics = {
        "dataset": cfg.app.dataset,
        "latent_dim": int(latent),
        "summaries": {
            "clean": clean_summary,
        },
        "samples": saved_aligned_samples,
    }
    if noise_summary is not None:
        aligned_metrics["summaries"]["noise"] = noise_summary

    aligned_metrics_path = reconstructed_text_path / "metrics.json"
    with open(aligned_metrics_path, "w", encoding="utf-8") as f_out:
        json.dump(aligned_metrics, f_out, indent=4, ensure_ascii=False)

    click.echo(
        f"[INFO] Reconstruction metrics | clean cosine: {clean_summary['mean_cosine']:.4f} "
        f"| clean l2: {clean_summary['mean_l2']:.4f} "
        f"| clean edit distance: {clean_summary['mean_edit_distance']:.4f}"
    )
    if noise_summary is not None:
        click.echo(
            f"[INFO] Reconstruction metrics | noise cosine: {noise_summary['mean_cosine']:.4f} "
            f"| noise l2: {noise_summary['mean_l2']:.4f} "
            f"| noise edit distance: {noise_summary['mean_edit_distance']:.4f}"
        )
    click.echo(f"[INFO] Aligned reconstructed texts saved to: {aligned_samples_path}")
    click.echo(f"[INFO] Aligned reconstruction metrics saved to: {aligned_metrics_path}")

def get_text_embeddings(cfg, device, experiments_root):
    click.echo(f"[INFO] Loading dataset: {cfg.app.dataset}")

    dataset = TextDatasetWrapper(
        dataset_name=cfg.app.dataset,
        column=cfg.app.column,
        seed=cfg.app.seed,
    )

    latent = cfg.neural.latent_dim[0] if isinstance(cfg.neural.latent_dim, ListConfig) else int(cfg.neural.latent_dim)
    path = get_experiment_path(cfg, experiments_root, latent)
    file_embeddings_path = path / "embeddings.pth"
    model = create_sonar_model(device)

    build_embeddings(
        model,
        dataset,
        modality=cfg.app.modality,
        device=device,
        save_path=file_embeddings_path,
    )

    if cfg.app.noise is not None and cfg.app.noise > 0:
        file_embeddings_path = path / "embeddings_noised.pth"
        noise_level = normalize_noise_level(get_scalar_config_value(cfg.app.noise))
        noised_train_texts = [
            apply_text_noise(text, noise_level=noise_level)
            for text in (dataset.train[index] for index in range(len(dataset.train)))
        ]
        noised_test_texts = [
            apply_text_noise(text, noise_level=noise_level)
            for text in (dataset.test[index] for index in range(len(dataset.test)))
        ]
        noised_dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
            column=cfg.app.column,
            seed=cfg.app.seed,
        )
        noised_dataset.train.texts = noised_train_texts
        noised_dataset.test.texts = noised_test_texts
        # Save the materialized test texts with noise for later analysis
        noised_test_texts_path = path / "noised_test_texts.npy"
        np.save(noised_test_texts_path, np.array(noised_test_texts, dtype=object))
        
        build_embeddings(
            model,
            noised_dataset,
            modality=cfg.app.modality,
            device=device,
            save_path=file_embeddings_path,
        )

def create_text_memories(cfg, n_saved, device, use_noise=False, experiments_root=None):
    latent = int(get_scalar_config_value(cfg.neural.latent_dim))
    domain = int(get_scalar_config_value(cfg.memory.domain))
    batch_size = get_memory_batch_size(cfg, domain)
    filling_percent = float(get_scalar_config_value(cfg.memory.filling))
    sigma = float(get_scalar_config_value(cfg.memory.sigma))
    iota = float(get_scalar_config_value(cfg.memory.iota))
    kappa = float(get_scalar_config_value(cfg.memory.kappa))
    xi = float(get_scalar_config_value(cfg.memory.xi))

    click.echo(
        f"[INFO] Creating text memories with latent={latent}, domain={domain}"
    )

    path = get_experiment_path(cfg, experiments_root, latent)
    embeddings_dataset = load_embeddings_dataset(path, device=device)

    dataset = TextDatasetWrapper(
        dataset_name=cfg.app.dataset,
        column=cfg.app.column,
        seed=cfg.app.seed,
    )

    all = torch.cat([embeddings_dataset.train.data, embeddings_dataset.test.data], dim=0)

    if use_noise:
        noised_embeddings_dataset = load_embeddings_dataset(path, device=device, noised=True)
        noised_test_texts = np.load(
            path / "noised_test_texts.npy",
            allow_pickle=True,
        ).tolist()

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
        batch_size=batch_size,
    )

    if use_noise:
        eval_dataset = noised_embeddings_dataset.test
    else:
        eval_dataset = embeddings_dataset.test
    
    (
        memories_features,
        recognitions,
        weights,
        recognized_rate,
        unrecognized_rate,
    ) = evalm_text(
        eam,
        dataset=eval_dataset,
        quantizer=quantizer,
        batch_size=batch_size,
    )

    total = len(memories_features)

    transformer = create_sonar_model(device)

    if use_noise:
        click.echo(f"[INFO] Evaluating text reconstruction with noise level: {cfg.app.noise}")
        output_path = ensure_directory(path / f"dim{domain}_memory_reconstructed_noise_{cfg.app.noise}")
    else:
        click.echo("[INFO] Evaluating text reconstruction without noise")
        output_path = ensure_directory(path / f"dim{domain}_memory_reconstructed")

    all_texts = [str(text) for text in dataset.test.texts[:total]]
    recognized_indices = np.flatnonzero(recognitions)
    reconstructable_texts = [all_texts[index] for index in recognized_indices]
    reconstructable_cues = (
        [str(noised_test_texts[index]) for index in recognized_indices]
        if use_noise
        else reconstructable_texts
    )
    reconstructable_embeddings = memories_features[recognitions]

    # Get metrics from recalled texts
    samples, summary = text_reconstruction_metrics(
        model=transformer,
        device=device,
        original_texts=reconstructable_texts,
        cue_texts=reconstructable_cues,
        embeddings=reconstructable_embeddings,
    )

    for sample, original_index in zip(samples, recognized_indices):
        sample["index"] = int(original_index)
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
        }
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
    model,
    device,
    original_texts,
    embeddings,
    cue_texts=None,
    batch_size=32,
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

    original_texts = [str(text) for text in original_texts]
    cue_texts = original_texts if cue_texts is None else [str(text) for text in cue_texts]

    if len(cue_texts) != len(original_texts):
        raise ValueError("cue_texts and original_texts must have the same length")

    if getattr(device, "type", None) == "mps":
        device = torch.device("cpu")
        target_device = device
    else:
        target_device = device

    source_embeddings = torch.as_tensor(
        embeddings,
        dtype=torch.float32,
        device=target_device,
    )

    reconstructed_texts = decode_text_embeddings(
        model=model,
        embeddings=source_embeddings,
        device=target_device,
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

    source_norm = torch.linalg.norm(source_embeddings, dim=1) # Compute norms for cosine similarity
    reconstructed_norm = torch.linalg.norm(reconstructed_embeddings, dim=1) # Compute norms for cosine similarity
    denom = torch.clamp(source_norm * reconstructed_norm, min=1e-12) # Avoid division by zero in cosine similarity calculation
    cosine_scores = (
        torch.sum(source_embeddings * reconstructed_embeddings, dim=1) / denom
    ) # Get cosine similarity scores between source and reconstructed embeddings
    l2_scores = torch.linalg.norm(source_embeddings - reconstructed_embeddings, dim=1) # Get L2 distance scores between source and reconstructed embeddings
    euclidean_scores = torch.sqrt(torch.sum((source_embeddings - reconstructed_embeddings) ** 2, dim=1)) # Get Euclidean distance scores between source and reconstructed embeddings

    samples = []
    for index, (original, cue, reconstructed, cosine, l2, euclidean) in enumerate(
        zip(original_texts, cue_texts, reconstructed_texts, cosine_scores, l2_scores, euclidean_scores)
    ):
        samples.append(
            {
                "index": index,
                "original": original,
                "cue": cue,
                "reconstructed": reconstructed,
                "cosine": float(cosine.item()),
                "euclidean": float(euclidean.item()),
                "l2": float(l2.item()),
                "edit_distance": int(
                    edit_distance(original.lower(), reconstructed.lower())
                ),
            }
        )

    summary = {
        "samples": len(samples),
        "mean_cosine": float(np.mean([item["cosine"] for item in samples])),
        "mean_euclidean": float(np.mean([item["euclidean"] for item in samples])),
        "mean_l2": float(np.mean([item["l2"] for item in samples])),
        "mean_edit_distance": float(
            np.mean([item["edit_distance"] for item in samples])
        ),
    }

    return samples, summary


def get_besttext_params(cfg, config, device, EXPERIMENTS_ROOT, use_noise=False):
    from omegaconf import ListConfig

    global_results = []
    noise_suffix = f"_noise_{cfg.app.noise}" if use_noise else ""

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
        if use_noise:
            noised_embeddings_dataset = load_embeddings_dataset(
                path,
                device,
                noised=True,
            )

        dataset = TextDatasetWrapper(
            dataset_name=cfg.app.dataset,
            column=cfg.app.column,
            seed=cfg.app.seed,
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
                                batch_size = get_memory_batch_size(cfg, msize)
                                
                                # Create memory
                                eam = AssociativeMemory(
                                    n=latent,
                                    m=msize,
                                    sigma=sigma,
                                    xi=xi,
                                    iota=iota,
                                    kappa=kappa,
                                )
                                # Fill memory with all embeddings (train + test) 
                                eam = memorize(
                                    eam,
                                    dataset=all_embeddings,
                                    quantizer=quantizer,
                                    filling_percent=filling_percent,
                                    batch_size=batch_size,
                                )
                                (
                                    memories_features,
                                    recognitions,
                                    _weights,
                                    recognized,
                                    unrecognized,
                                ) = evalm_text(
                                    eam,
                                    dataset=(
                                        noised_embeddings_dataset.test
                                        if use_noise
                                        else embeddings_dataset.test
                                    ),
                                    quantizer=quantizer,
                                    batch_size=batch_size,
                                )

                                recognized_embeddings = memories_features[recognitions]
                                recognized_texts = [
                                    text
                                    for text, is_recognized in zip(test_texts, recognitions)
                                    if is_recognized
                                ]


                                # Obtain reconstruction metrics for the recognized texts and their corresponding embeddings
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
                                        "mean_euclidean": summary["mean_euclidean"],
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

            path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
            save_path = path / f"{latent}_parameters_search_results{noise_suffix}.partial.json"
            click.echo(f"[INFO] Saving results to: {save_path}")
            with open(save_path, "w") as f:
                json.dump(global_results, f, indent=4)

        progress.reset(msize_task)
        global_results.extend(results)
        progress.update(
            latent_task, advance=1, description=f"[magenta]Latent: {latent}"
        )

    progress.stop()

    path = get_experiment_path(cfg, EXPERIMENTS_ROOT, latent)
    save_path = path / f"global_memories_results{noise_suffix}.json"
    click.echo(f"[INFO] Saving results to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(global_results, f, indent=4)

    df = pd.DataFrame(global_results)
    df = df.sort_values(
        by=["mean_cosine", "recognized", "mean_edit_distance", "mean_l2"],
        ascending=[False, False, True, True],
    )

    best_params = df.iloc[0].to_dict()
    best_domain = int(best_params["msize"])
    best_batch_size = get_memory_batch_size(cfg, best_domain)
    cfg.neural.latent_dim = [int(best_params["latent"])]
    cfg.memory.domain = [best_domain]
    cfg.memory.batch_size = best_batch_size
    cfg.memory.filling = [float(best_params["filling_percent"])]
    cfg.memory.sigma = float(best_params["sigma"])
    cfg.memory.xi = float(best_params["xi"])
    cfg.memory.iota = float(best_params["iota"])
    cfg.memory.kappa = float(best_params["kappa"])

    config_path = Path(config)
    best_config_path = config_path.with_name(
        f"{config_path.stem}.best{noise_suffix}{config_path.suffix}"
    )
    click.echo(
        f"[INFO] Saving updated config with best parameters to: {best_config_path}"
    )
    with open(best_config_path, "w") as f:
        OmegaConf.save(cfg, f)

    click.echo("[INFO] Best text parameters search completed.")


def interactive_memory(cfg, device, experiments_root):
    import gradio as gr

    latent = int(get_scalar_config_value(cfg.neural.latent_dim))
    domain = int(get_scalar_config_value(cfg.memory.domain))
    batch_size = get_memory_batch_size(cfg, domain)
    filling_percent = float(get_scalar_config_value(cfg.memory.filling))
    sigma = float(get_scalar_config_value(cfg.memory.sigma))
    iota = float(get_scalar_config_value(cfg.memory.iota))
    kappa = float(get_scalar_config_value(cfg.memory.kappa))
    xi = float(get_scalar_config_value(cfg.memory.xi))

    click.echo(f"[INFO] Creating UI with latent={latent}, domain={domain}")

    path = get_experiment_path(cfg, experiments_root, latent)
    embeddings_dataset = load_embeddings_dataset(path, device=device)

    all = torch.cat([embeddings_dataset.train.data, embeddings_dataset.test.data], dim=0)

    quantizer = Quant(all)
    transformer = create_sonar_model(device)

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
        batch_size=batch_size,
    )

    def update_memory_params(eam_instance, xi_value, sigma_value, iota_value, kappa_value):
        if eam_instance is None:
            return None, "La memoria no esta inicializada."

        eam_instance.xi = int(xi_value)
        eam_instance.sigma = float(sigma_value)
        eam_instance.iota = float(iota_value)
        eam_instance.kappa = float(kappa_value)

        return (
            eam_instance,
            "Parametros actualizados en la instancia actual: "
            f"xi={eam_instance.xi}, sigma={eam_instance.sigma:.4f}, "
            f"iota={eam_instance.iota:.4f}, kappa={eam_instance.kappa:.4f}."
        )

    def apply_noise_to_cue(cue):
        cue = str(cue).strip()
        if not cue:
            return ""

        return apply_text_noise(cue, noise_level=get_scalar_config_value(cfg.app.noise or 0.5))

    def interactive_recall(cue, eam_instance, progress=gr.Progress()):
        cue = str(cue).strip()
        if not cue:
            return "", "Escribe un texto para consultar la memoria."

        if eam_instance is None:
            return "", "La memoria no esta inicializada."

        progress(0.1, desc="Codificando entrada")
        with torch.no_grad():
            cue_embedding = transformer.encode([cue], device=device)
            cue_embedding = cue_embedding.cpu().numpy()

            progress(0.35, desc="Cuantizando embedding")
            cue_quantized = quantizer.quantize(cue_embedding, eam_instance.m)

            progress(0.6, desc="Consultando memoria")
            recalled_embeddings, recognized, weights = eam_instance.batch_recall(cue_quantized)
            recognized = bool(np.asarray(recognized)[0])
            weight = float(np.asarray(weights, dtype=float)[0])

            if not recognized:
                progress(1.0, desc="Sin coincidencia")
                return "", f"No se reconocio ninguna memoria para la entrada. Peso medio: {weight:.4f}."

            progress(0.8, desc="Reconstruyendo embedding")
            recalled_embedding = quantizer.dequantize(
                np.asarray(recalled_embeddings, dtype=float), eam_instance.m
            )[0]
            recalled_embedding = torch.as_tensor(
                recalled_embedding, dtype=torch.float32
            ).unsqueeze(0)

            progress(0.95, desc="Decodificando texto")
            recalled_text = decode_text_embeddings(
                model=transformer,
                embeddings=recalled_embedding,
                device=device,
            )[0]

        progress(1.0, desc="Completado")
        return recalled_text, f"Memoria reconocida correctamente. Peso medio: {weight:.4f}."

    with gr.Blocks() as recall:
        gr.Markdown(
            f"## Memoria interactiva\nLatente: {latent} | Dominio: {domain} | Filling: {filling_percent:.2f}"
        )
        gr.Markdown(
            "Puedes actualizar xi, sigma, iota y kappa sobre la misma instancia de memoria. "
            "Si quieres cambiar domain, hay que reconstruir la memoria."
        )

        memory_state = gr.State(value=eam)

        with gr.Row():
            xi_input = gr.Number(value=xi, label="Xi", precision=0)
            sigma_input = gr.Number(value=sigma, label="Sigma")
            iota_input = gr.Number(value=iota, label="Iota")
            kappa_input = gr.Number(value=kappa, label="Kappa")

        update_button = gr.Button("Actualizar parametros", variant="primary")
        memory_status = gr.Markdown(
            value=(
                "Memoria inicializada con "
                f"xi={eam.xi}, sigma={eam.sigma:.4f}, iota={eam.iota:.4f}, kappa={eam.kappa:.4f}."
            )
        )

        cue_input = gr.Textbox(
            lines=3,
            label="Texto de entrada",
            placeholder="Escribe una frase para consultar la memoria",
        )
        noise_button = gr.Button("Aplicar ruido al texto")
        recall_button = gr.Button("Consultar memoria")
        recalled_output = gr.Textbox(label="Texto recordado")
        recall_status = gr.Markdown(label="Estado")

        update_button.click(
            fn=update_memory_params,
            inputs=[memory_state, xi_input, sigma_input, iota_input, kappa_input],
            outputs=[memory_state, memory_status],
        )

        noise_button.click(
            fn=apply_noise_to_cue,
            inputs=[cue_input],
            outputs=[cue_input],
        )

        recall_button.click(
            fn=interactive_recall,
            inputs=[cue_input, memory_state],
            outputs=[recalled_output, recall_status],
        )

        cue_input.submit(
            fn=interactive_recall,
            inputs=[cue_input, memory_state],
            outputs=[recalled_output, recall_status],
        )

    recall.launch()