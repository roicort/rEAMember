import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


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
            embeddings.append(z.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def get_embeddings( model, dataset, device, modality='image', save_path=None, batch_size=64, num_workers=2):
    """
    Extrae embeddings usando el encoder y un dataset (no dataloader).
    Crea internamente el DataLoader para asegurar el batch correcto.
    """
    model.eval()
    if device is not None:
        model.to(device)

    dataloader_train = DataLoader(dataset.train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_test = DataLoader(dataset.test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if modality == 'text':
        pass
        #embeddings_train = evalText(dataloader_train, model, device)
        #embeddings_test = evalText(dataloader_test, model, device)

    elif modality == 'image':
        embeddings_train = evalImage(dataloader_train, model, device)
        embeddings_test = evalImage(dataloader_test, model, device)

    print("[INFO] Embeddings train shape:", embeddings_train.shape)
    print("[INFO] Embeddings test shape:", embeddings_test.shape)

    from .dataset import EmbeddingDatasetWrapper

    embedding_dataset = EmbeddingDatasetWrapper(
        train=embeddings_train,
        test=embeddings_test,
        labels_train=dataset.train.targets,
        labels_test=dataset.test.targets
    )

    if save_path is not None:

        torch.save(embedding_dataset, save_path / 'embeddings.pth')

        try:
            dataset_name = getattr(getattr(dataset, 'train', None), '__class__', type('X', (), {}))
            dataset_name = getattr(dataset_name, '__name__', 'dataset')
            model_name = getattr(model, '__class__', type('Y', (), {}))
            model_name = getattr(model_name, '__name__', 'model')

            # The latent dimension is the second dimension of the embeddings
            latent = embedding_dataset.train.data.shape[1]

            tb_dir = Path('./logs') / f"{dataset_name}-{model_name}{latent}_embeddings"
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

            metadata_train = _labels_to_list_str(embedding_dataset.train.targets)
            metadata_test = _labels_to_list_str(embedding_dataset.test.targets)

            writer.add_embedding(emb_train, metadata=metadata_train, tag='train-embeddings')
            writer.add_embedding(emb_test, metadata=metadata_test, tag='test-embeddings')

            writer.flush()
            writer.close()
            print(f"[INFO] Embeddings registrados en TensorBoard: {tb_dir}")
        except Exception as e:
            print(f"[WARN] No se pudieron registrar embeddings en TensorBoard: {e}")

        # PCA 
        from sklearn.decomposition import PCA
        embeddings_np = torch.cat([embedding_dataset.train.data, embedding_dataset.test.data], dim=0).numpy()
        if embeddings_np.ndim > 2:
            embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(embeddings_np)

        # Plot PCA embeddings using plotly
        import plotly.express as px
        labels_all = torch.cat([embedding_dataset.train.targets, embedding_dataset.test.targets], dim=0).numpy()
        fig = px.scatter(x=pca_embeddings[:, 0], y=pca_embeddings[:, 1], color=labels_all, title='PCA Embeddings')
        fig.update_layout(width=800, height=600)
        fig.write_image(save_path / 'pca_embeddings.png')
