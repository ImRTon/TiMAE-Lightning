from lightning import LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import MLFlowLogger
import torch
from network.dataset import UCR2018, KaggleECGCategorization
from network.ti_mae import TiMAEForPretraining, TiMAE
import lightning.pytorch as pl
import os
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.strategies import DDPStrategy
# from network.ti_mae_mamba import TiMAEMambaForPretraining, TiMAEMamba

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import MLFlowLogger
import lightning.pytorch as pl
import os
from lightning.pytorch.callbacks import ModelCheckpoint


class MLFlowSaveConfigCallback(SaveConfigCallback):
    def __init__(self, parser, config, config_filename='config.yaml', overwrite=False, multifile=False):
        super().__init__(parser, config, config_filename, overwrite, multifile, save_to_log_dir=False)

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        config_dict = vars(self.config)
        pl_module.logger.log_hyperparams(config_dict)
        
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments('data.num_classes', 'model.init_args.num_classes', apply_on="instantiate")
        parser.link_arguments('data.class_weights', 'model.init_args.class_weights', apply_on="instantiate")
        parser.set_defaults({
            "trainer.strategy": "ddp_find_unused_parameters_true",
        })
    def configure_callbacks(self):
        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss", 
            save_top_k=1,
            mode="min",
            filename="best-{epoch:02d}-{val_loss:.4f}",
            save_last=True,
        )
        return super().configure_callbacks() + [checkpoint_cb]

def cli_compile_main():
    cli = MyLightningCLI(datamodule_class=UCR2018, 
                       save_config_kwargs={'overwrite': True}, 
                       run=False, 
                       save_config_callback=MLFlowSaveConfigCallback
                )
    compiled_model = cli.model
    cli.trainer.fit(compiled_model, datamodule=cli.datamodule)
    cli.trainer.test(datamodule=cli.datamodule)
if __name__ == '__main__':
    cli_compile_main()
