"""
This file runs the main training/val loop, etc... using Lightning Trainer"""
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from argparse import ArgumentParser

from cifar import BnnOnCIFAR10


def main(hparams):
    # init module
    model = BnnOnCIFAR10(hparams)

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath="../../results",
        save_best_only=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        # checkpoint_callback=checkpoint_callback
        default_save_path="../../results",
        check_val_every_n_epoch=,
        show_progress_bar=True,
    )

    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--max_nb_epochs", default=500, type=int)

    # give the module a chance to add own params
    # good practice to define LightningModule specific params in the module
    parser = BnnOnCIFAR10.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
