"""ECG beat classifiers, shared by ecg_classify_beats.ipynb and transfer_learning.ipynb.

Two architectures, selected by name ("arch"):

- "paper":  the residual CNN of Kachuee et al. (2018), Fig. 2 and section III-B.
- "simple": the original 4-convolution CNN, adapted from
            https://github.com/sajiddeboss/ECG-HeartBeat-Classification

Both expose the same interface for transfer learning:
    model.features(x)       -> (batch, model.feature_dim) beat representation
    model.feature_layers()  -> the modules that compute it (unfrozen when fine-tuning)

The models live in a module (not in the notebooks) so that both notebooks build
exactly the same classes, which checkpoints need in order to load.
"""
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from torch.optim import lr_scheduler

BEAT_LENGTH = 187  # samples per beat in the Kaggle MIT-BIH and PTB CSVs


class EcgClassifier(nn.Module):
    """The "simple" architecture: four plain convolutions and a large dense head."""

    feature_dim = 256

    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU()

    def features(self, x):
        """Last convolution layer, globally average-pooled: 256 values per beat."""
        # Input shape: (batch, 187) -> add channel dim -> (batch, 1, 187)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return self.global_avg_pool(x).flatten(1)

    def feature_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, x):
        x = self.dropout(self.features(x))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)  # logits; CrossEntropyLoss applies the softmax


class ResidualBlock(nn.Module):
    """Conv -> ReLU -> Conv, plus a skip connection, -> ReLU -> max-pool (Fig. 2)."""

    def __init__(self, channels: int = 32, kernel_size: int = 5):
        super().__init__()
        # "same" padding keeps the length unchanged so the skip connection can be added.
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv2(self.relu(self.conv1(x)))
        return self.pool(self.relu(x + y))


class ResidualEcgClassifier(nn.Module):
    """The "paper" architecture (Kachuee et al. 2018, Fig. 2): a convolution, five
    residual blocks of 32 kernels of size 5 with max-pooling (size 5, stride 2), then
    two fully connected layers. 13 weight layers, about 54k parameters."""

    def __init__(self, num_classes: int = 5, channels: int = 32, n_blocks: int = 5):
        super().__init__()
        self.stem = nn.Conv1d(1, channels, kernel_size=5, padding="same")
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        length = BEAT_LENGTH
        for _ in range(n_blocks):
            length = (length - 5) // 2 + 1  # 187 -> 92 -> 44 -> 20 -> 8 -> 2
        self.feature_dim = channels * length  # 32 channels x 2 positions = 64

        self.fc1 = nn.Linear(self.feature_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def features(self, x):
        """Output activations of the last convolution block, flattened (the paper's
        representation for transfer learning): 64 values per beat."""
        x = self.blocks(self.stem(x.unsqueeze(1)))
        return x.flatten(1)

    def feature_layers(self):
        return [self.stem, self.blocks]

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(self.features(x))))  # logits


def make_ecg_model(arch: str, num_classes: int = 5) -> nn.Module:
    if arch == "paper":
        return ResidualEcgClassifier(num_classes)
    if arch == "simple":
        return EcgClassifier(num_classes)
    raise ValueError(f"Unknown arch {arch!r}: use 'paper' or 'simple'")


class EcgLightningModule(pl.LightningModule):
    """Trains an arrhythmia classifier. The training recipe follows the architecture:

    - "paper":  Adam (lr 1e-3, betas 0.9/0.999), learning rate multiplied by 0.75
                every 10,000 steps (section III-D).
    - "simple": AdamW with ReduceLROnPlateau on validation loss.

    Checkpoints saved before `arch` existed have no hyperparameters; they load
    with the defaults, i.e. as the "simple" architecture.
    """

    def __init__(self, num_classes: int = 5, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 class_weights: torch.Tensor = None, arch: str = "simple"):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.arch = arch
        self.model = make_ecg_model(arch, num_classes)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # class_weights will be moved to the correct device automatically via register_buffer
        self.register_buffer("class_weights", class_weights)

        # Epoch-averaged history (populated in on_*_epoch_end hooks)
        self.training_step_outputs = {"loss": [], "acc": []}
        self.validation_step_outputs = {"loss": [], "acc": []}

        # torchmetrics accumulate correctly across all batches in an epoch
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.classification.Accuracy(num_classes=num_classes, task="multiclass")
        self.val_acc = torchmetrics.classification.Accuracy(num_classes=num_classes, task="multiclass")

    @property
    def loss_fn(self):
        return nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_loss(loss)
        self.train_acc(logits, y)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_loss(loss)
        self.val_acc(logits, y)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        self.training_step_outputs["loss"].append(metrics["train_loss_epoch"].item())
        self.training_step_outputs["acc"].append(metrics["train_acc_epoch"].item())

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            self.validation_step_outputs["loss"].append(metrics["val_loss"].item())
            self.validation_step_outputs["acc"].append(metrics["val_acc"].item())

    def configure_optimizers(self):
        if self.arch == "paper":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
            scheduler = lr_scheduler.LambdaLR(optimizer, lambda step: 0.75 ** (step // 10_000))
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",       # minimise val_loss
            factor=0.5,       # halve the LR on a plateau
            patience=2,       # after 2 epochs without improvement
            min_lr=1e-6,      # don't reduce below 1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",   # required for ReduceLROnPlateau
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x).argmax(dim=1)
