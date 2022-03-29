import numpy as np
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from det_models.coco_eval import CocoEvaluator
from det_models.coco_utils import get_coco_api_from_dataset

import mlflow
import mlflow.pytorch

# from clf_models.model import POIClassifier
from det_models.model import POIDetection
# from datasets.datamodule import POIDataModule

from datasets_signboard_detection.datamodule import POIDataModule
MLFLOW_TRACKING_URI = "databricks"
MLFLOW_EXPERIMENT = "/Users/hunglv@piv.asia/Mattress_Detection"


def _parse_args():
    parser = argparse.ArgumentParser(description="POI classification based on image")
    # model parameters
    parser.add_argument('--train-path', type=str, default=None,
                        help='Path to training data')
    parser.add_argument('--test-path', type=str, default=None,
                        help='Path to testing data')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to image folder')
    parser.add_argument('--label-path', type=str, default=None,
                        help='Path to label folder')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--backbone', type=str, default="maskrcnn_resnet50_fpn",
                        help='backbone for model')
    parser.add_argument('--tune-fc-only', action='store_true',
                        help='tuning last fc layer only')

    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--optimizer', type=str, default="adamw",
                        help='optimizer name',
                        choices=["adamw", "sgd"])
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        metavar='M', help='w-decay')
    # optimizer params if SGD
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # optimizer params if AdamW
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps')

    # cuda, seed and logging
    parser.add_argument('--gpu-ids', default=0, type=int, nargs="+",
                        help='use which gpu to train (default=0)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    # name
    parser.add_argument('--resume', type=str, default=None, 
                        help='resume training')

    args = parser.parse_args()
    return args


def handle_train(args):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    
    dm = POIDataModule(
                    train_path=args.train_path,
                    test_path=None,
                    data_path=args.data_path,
                    label_path=args.label_path,
                    train_batch_size=args.train_batch_size,
                    test_batch_size=args.test_batch_size,
                    seed=args.seed)
    dm.setup(stage="fit")
    n_classes = 3

    coco = get_coco_api_from_dataset(dm.val_dataloader().dataset)
    iou_types = ["bbox","segm"]
    # iou_types.append("segm")
    coco_evaluator = CocoEvaluator(coco, iou_types)
    
    model = POIDetection(n_classes=n_classes,
                          **vars(args))
    model.setup_evaluator(dm.val_dataloader().dataset)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss', # monitored quantity
        filename='Sequence-{epoch:02d}-{val_loss:.5f}',
        save_top_k=3, # save the top 3 models
        mode='min', # mode of the monitored quantity for optimization
    )
    
    early_stop_callback = EarlyStopping(
        monitor="train_loss", 
        min_delta=0.00, patience=3, 
        verbose=False, 
        mode="min"
    )
    # callbacks=[checkpoint_callback, early_stop_callback]
    callbacks=[checkpoint_callback]

    mlflow.pytorch.autolog()
    # use float (e.g 1.0) to set val frequency in epoch
    # if val_check_interval is integer, val frequency is in batch step
    training_params = {
        "callbacks": callbacks,
        "gpus": 2,
        "val_check_interval": 1.0,
        "max_epochs": args.epochs,
        "strategy": "dp"
    }
    if args.resume is not None:
        training_params["resume_from_checkpoint"] = args.resume
    
    trainer = pl.Trainer(**training_params)
    trainer.fit(model, dm)

def main():
    args = _parse_args()
    handle_train(args)
    # handle_predict(args)


if __name__ == "__main__":
    main()