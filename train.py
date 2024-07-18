from lightning.pytorch.cli import LightningCLI

from network.dataset import UCR2018
from network.ti_mae import TiMAEForPretraining, TiMAE

def main():
    cli = LightningCLI(
        datamodule_class=UCR2018,
        save_config_kwargs={'overwrite': True}
    )

if __name__ == '__main__':
    main()