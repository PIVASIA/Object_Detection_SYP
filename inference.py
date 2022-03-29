import numpy as np
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

import mlflow
import mlflow.pytorch

from det_models.model import POIDetection
from datasets_detection.datamodule import POIDataModule

def _parse_args():
    parser = argparse.ArgumentParser(description="POI classification based on image")
    # model parameters

    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to testing data')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to data folder')
    parser.add_argument('--label-path', type=str, default=None,
                        help='Path to label folder')                 
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to output of image')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to checkpoint')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    return args

def load_model(checkpoint_path):
    model = POIDetection.load_from_checkpoint(checkpoint_path=checkpoint_path)
    return model

def handle_inference(args):

    dm = POIDataModule(
            train_path=None,
            test_path=args.test_path,
            data_path=args.data_path,
            label_path=None,
            seed=args.seed)
    dm.setup("predict")

    model = load_model(args.checkpoint_path)

    from det_models.inference_detection import POIDetectionTask
    task = POIDetectionTask(model,
                            output_path = args.output_path,
                            test_path=args.test_path,
                            data_path=args.data_path)

    trainer = pl.Trainer(gpus=0)
    trainer.predict(task, datamodule=dm)


def main():
    args = _parse_args()
    handle_inference(args)

if __name__ == "__main__":
    main()