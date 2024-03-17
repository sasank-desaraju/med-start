import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Make a Pytorch Lightning Module for ChestMNIST using a CNN
class ChestMNISTLitModel(pl.LightningModule):
    def __init__(self, channels=[32, 64, 128, 256], kernel_size=3, pool_size=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.channels = channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = pl.metrics.Accuracy()
        
        # convolutional layers
        self.convs = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(channels[0], channels[1], kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(channels[1], channels[2], kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(channels[2], channels[3], kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
        )
        
        # dense layers
        self.linear_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 14),
        )
        
    def forward(self, x):
        # called with self(x)
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(preds, y), prog_bar=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.accuracy(preds, y), prog_bar=True)
        return loss