"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import os
import sys
from pytorch_lightning import Trainer
from argparse import ArgumentParser

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "../../..")
sys.path.append(ROOT)

from src.cnn1d.baseline.baseline_model import HARCNN


def main(hparams):
    # init module
    model = HARCNN(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_epochs=hparams.max_nb_epochs,
        early_stop_callback=False,
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = HARCNN.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()
    print(hparams)

    main(hparams)
