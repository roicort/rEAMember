import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval(dataloader, autoencoder, device=None):
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
            else:
                x = batch
                y = None
            x = x.to(device)
            z = autoencoder.encode(x)
            embeddings.append(z.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def get_embeddings(autoencoder, dataset, device, save_path=None, batch_size=64, num_workers=2):
    """
    Extrae embeddings usando el encoder y un dataset (no dataloader).
    Crea internamente el DataLoader para asegurar el batch correcto.
    """
    autoencoder.eval()
    if device is not None:
        autoencoder.to(device)

    dataloader_train = DataLoader(dataset.train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_test = DataLoader(dataset.test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embeddings_train = eval(dataloader_train, autoencoder, device)
    embeddings_test = eval(dataloader_test, autoencoder, device)

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
