import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

class Autoencoder(pl.LightningModule):
    def __init__(self, input_shape, latent_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        c, h, w = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear((h//4)*(w//4)*32, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, (h//4)*(w//4)*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, h//4, w//4)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, c, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.HuberLoss()
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = self.train_mae(x_hat, x)
        self.log('train_loss', loss)
        self.log('train_mae', mae)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = self.val_mae(x_hat, x)
        self.log('val_loss', loss)
        self.log('val_mae', mae)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
