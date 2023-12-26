# Create Lightning Module for our ChestMNIST dataset using MONAI UNet, loss, and metrics
import lightning as L
import torch
import monai

class ChestMNISTLitModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss = monai.losses.DiceLoss(sigmoid=True)
        self.accuracy = monai.metrics.DiceMetric(include_background=False, reduction="mean")
        
        # UNet model
        self.model = monai.networks.nets.UNet(
            dimensions=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        
    def forward(self, x):
        # called with self(x)
        return self.model(x)
    
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
        self.log('val_loss', loss)
        self.accuracy(y_pred=logits, y=y)
        self.log('val_dice', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('test_loss', loss)
        self.accuracy(y_pred=logits, y=y)
        self.log('test_dice', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss