from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import plotly.express as px
from sklearn.decomposition import PCA

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
    save_path=Path("embeddings.pth"),
    batch_size=32,
    num_workers=2,
):
    """
    Extrae embeddings usando el encoder y un dataset (no dataloader).
    Crea internamente el DataLoader para asegurar el batch correcto.
    """

    if not os.path.exists(save_path):
        dataloader_train = DataLoader(
            dataset.train, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        dataloader_test = DataLoader(
            dataset.test, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        print(f"[INFO] Not found embeddings at: {save_path}, extracting...")

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

        from .datasets.embedding import EmbeddingDatasetWrapper

        embedding_dataset = EmbeddingDatasetWrapper(
            train=embeddings_train,
            test=embeddings_test,
            labels_train=dataset.train.targets,
            labels_test=dataset.test.targets,
        )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Saving embeddings in: {save_path}")
            torch.save(embedding_dataset, save_path)

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

    else:
        print(f"[INFO] Embeddings already exist: {save_path}, plotting...")


    # Load embeddings with labels for plotting
    from reamember.datasets.embedding import EmbeddingDatasetWrapper, EmbeddingDataset
    torch.serialization.add_safe_globals([EmbeddingDatasetWrapper, EmbeddingDataset])
    embedding_dataset = torch.load(save_path)

    # Plot PCA embeddings using plotly

    pca = PCA(n_components=2)
    mebeddings_all = torch.cat([embedding_dataset.train.data, embedding_dataset.test.data], dim=0).cpu().numpy()
    pca_embeddings = pca.fit_transform(mebeddings_all)

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
    fig.update_layout(width=3840, height=2160)
    save_path = Path(save_path)
    fig.write_html(str(save_path.with_name(f"{save_path.stem}_pca.html")))
    fig.write_image(str(save_path.with_name(f"{save_path.stem}_pca.png")))
