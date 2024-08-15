from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from network.dataset import UCR2018, KaggleECGCategorization
from network.ti_mae import TiMAEForPretraining, TiMAE
from network.ti_mae_mamba import TiMAEMambaForPretraining, TiMAEMamba

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments('data.num_classes', 'model.init_args.num_classes', apply_on="instantiate")
        parser.link_arguments('data.class_weights', 'model.init_args.class_weights', apply_on="instantiate")

def main():
    cli = MyLightningCLI(
        datamodule_class=UCR2018,
        save_config_kwargs={'overwrite': True}
    )

if __name__ == '__main__':
    main()