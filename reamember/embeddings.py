from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def evalImage(dataloader, model, device=None):
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, _ = batch
            else:
                x = batch
            x = x.to(device)
            z = model.encode(x)
            embeddings.append(z)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def evalText(dataloader, model, device=None):
    embeddings = []
    latent_dim = getattr(model, "latent_dim", 0)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                texts, _ = batch
            else:
                texts = batch

            if isinstance(texts, str):
                texts = [texts]
            elif not isinstance(texts, list):
                texts = list(texts)

            batch_embeddings = model.encode(
                texts,
                source_lang="eng_Latn",
                max_seq_len=128,
                device=device,
            )

            if torch.is_tensor(batch_embeddings):
                embeddings.append(batch_embeddings.detach().cpu().float())
            else:
                embeddings.append(torch.as_tensor(batch_embeddings, dtype=torch.float32))

    if len(embeddings) == 0:
        return torch.empty((0, latent_dim), dtype=torch.float32)

    return torch.cat(embeddings, dim=0)


def get_embeddings(
    model,
    dataset,
    device,
    modality="image",
    save_path=None,
    batch_size=32,
    num_workers=2,
):
    """
    Extrae embeddings usando el encoder y un dataset (no dataloader).
    Crea internamente el DataLoader para asegurar el batch correcto.
    """

    dataloader_train = DataLoader(
        dataset.train, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    dataloader_test = DataLoader(
        dataset.test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if modality == "text":
        model.eval()
        if device is not None:
            model.to(device)
        embeddings_train = evalText(dataloader_train, model, device)
        embeddings_test = evalText(dataloader_test, model, device)

    elif modality == "image":
        model.eval()
        if device is not None:
            model.to(device)
        embeddings_train = evalImage(dataloader_train, model, device)
        embeddings_test = evalImage(dataloader_test, model, device)

    print("[INFO] Embeddings train shape:", embeddings_train.shape)
    print("[INFO] Embeddings test shape:", embeddings_test.shape)

    from .dataset import EmbeddingDatasetWrapper

    embedding_dataset = EmbeddingDatasetWrapper(
        train=embeddings_train,
        test=embeddings_test,
        labels_train=dataset.train.targets,
        labels_test=dataset.test.targets,
    )

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving embeddings in: {save_path}")
        torch.save(embedding_dataset, save_path / "embeddings.pth")

        try:
            dataset_name = getattr(
                getattr(dataset, "train", None), "__class__", type("X", (), {})
            )
            dataset_name = getattr(dataset_name, "__name__", "dataset")
            model_name = getattr(model, "__class__", type("Y", (), {}))
            model_name = getattr(model_name, "__name__", "model")

            # The latent dimension is the second dimension of the embeddings
            latent = embedding_dataset.train.data.shape[1]

            tb_dir = Path("./logs") / f"{dataset_name}_{model_name}_{latent}_embeddings"
            writer = SummaryWriter(log_dir=str(tb_dir))
            emb_train = embeddings_train
            emb_test = embeddings_test

            # Prepare metadata (labels) as list[str]
            def _labels_to_list_str(labels):
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().tolist()
                try:
                    return [str(int(val)) for val in labels]
                except Exception:
                    return [str(val) for val in labels]

            labels_train = _labels_to_list_str(embedding_dataset.train.targets)
            labels_test = _labels_to_list_str(embedding_dataset.test.targets)

            writer.add_embedding(
                emb_train, metadata=labels_train, tag="train-embeddings"
            )
            writer.add_embedding(emb_test, metadata=labels_test, tag="test-embeddings")
            writer.close()

            print(f"[INFO] Embeddings registrados en TensorBoard: {tb_dir}")
        except Exception as e:
            print(f"[WARN] No se pudieron registrar embeddings en TensorBoard: {e}")

        # PCA
        from sklearn.decomposition import PCA

        embeddings_np = (
            torch.cat(
                [embedding_dataset.train.data, embedding_dataset.test.data], dim=0
            )
            .cpu()
            .numpy()
        )
        if embeddings_np.ndim > 2:
            embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(embeddings_np)

        # Load embeddings with labels for plotting
        embedding_dataset = torch.load(save_path / "embeddings.pth", map_location="cpu")

        # Plot PCA embeddings using plotly
        import plotly.express as px

        if embedding_dataset.train.targets is not None and embedding_dataset.test.targets is not None:
            labels_all = (
                torch.cat(
                    [embedding_dataset.train.targets, embedding_dataset.test.targets], dim=0
                )
                .cpu()
                .numpy()
            )
        else:
            labels_all = ["train"] * len(embedding_dataset.train.data) + ["test"] * len(
                embedding_dataset.test.data
            )
        fig = px.scatter(
            x=pca_embeddings[:, 0],
            y=pca_embeddings[:, 1],
            color=labels_all,
            title="PCA Embeddings",
        )
        fig.update_layout(width=800, height=600)
        fig.write_image(save_path / "embeddings_pca.png")
