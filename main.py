from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

from scripts.datamodule import ChestMNISTDataModule
from scripts.monai_net import ChestMNISTLitModel

def cli_main():
    #cli = LightningCLI(DemoModel, BoringDataModule, seed_everything_default=42)
    cli = LightningCLI(ChestMNISTLitModel, ChestMNISTDataModule, seed_everything_default=42)
    #cli.trainer.tune(cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    cli_main()