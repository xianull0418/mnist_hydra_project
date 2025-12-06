import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import Accuracy


class SimpleClassifier(pl.LightningModule):
    def __init__(self, in_features, hidden_dim, out_features, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=out_features)

        # 定义简单的三层网络
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        # MNIST 图片是 [batch, 1, 28, 28]，展平成 [batch, 784]
        x = x.view(x.size(0), -1)
        return self.net(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)