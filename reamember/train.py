import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from reamember.neuralnets.autoencoder import Autoencoder
from reamember.neuralnets.classifier import Classifier
import torchvision.utils as vutils

class ReconstructionsCallback(pl.Callback):
    def __init__(self, val_loader, logger, every_n_epochs=5, max_images=8):
        super().__init__()
        self.val_loader = val_loader
        self.logger = logger
        self.every_n_epochs = every_n_epochs
        self.max_images = max_images

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        pl_module.eval()
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            x, _ = batch
            x = x.to(pl_module.device)
            x_hat = pl_module(x)
            n = min(self.max_images, x.size(0))
            comparison = torch.cat([x[:n], x_hat[:n]])
            grid = vutils.make_grid(comparison, nrow=n, normalize=True, scale_each=True)
            trainer.logger.experiment.add_image('Reconstructions', grid, epoch)
        pl_module.train()

def train_autoencoder(config, input_shape, dataset, name, save_path):
    train_loader = DataLoader(
        dataset.train,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        dataset.test,
        batch_size=config.batch_size,
        num_workers=4,
        persistent_workers=True
    )
    model = Autoencoder(input_shape=input_shape, latent_dim=config.latent_dim)
    early_stop = EarlyStopping(monitor='val_loss', patience=config.patience, mode='min', verbose=True)
    logger = TensorBoardLogger("logs", name=f"{name}_autoencoder")
    recon_cb = ReconstructionsCallback(test_loader, logger)
    trainer = pl.Trainer(max_epochs=config.epochs, callbacks=[early_stop, recon_cb], logger=logger)
    trainer.fit(model, train_loader, test_loader)
    torch.save(model.state_dict(), save_path)

def train_classifier(config, dataset, name, save_path):
    train_loader = DataLoader(
        dataset.train,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True,
        persistent_workers=False 
    )
    test_loader = DataLoader(
        dataset.test,
        batch_size=config.batch_size,
        num_workers=0,
        persistent_workers=False
    )
    model = Classifier(latent_dim=config.latent_dim, n_classes=dataset.n_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=config.patience, mode='min', verbose=True)
    logger = TensorBoardLogger("logs", name=f"{name}_classifier")
    trainer = pl.Trainer(max_epochs=config.epochs, callbacks=[early_stop], logger=logger)
    trainer.fit(model, train_loader, test_loader)
    torch.save(model.state_dict(), save_path)
