# Clasificador en PyTorch (estructura base)
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(pl.LightningModule):
    def __init__(
        self,
        latent_dim,
        n_classes,
        hidden_dims=[256, 128, 128, 64],
        dropout=0.3,
        lr=1e-3,
        weight_decay=1e-4,
        scheduler_gamma=0.95
    ):
        super().__init__()
        layers = []
        input_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, n_classes))
        self.net = nn.Sequential(*layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
