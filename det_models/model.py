import torch
import torch.optim as optim
import pytorch_lightning as pl
from det_models.backbone import initialize_model
from det_models.coco_eval import CocoEvaluator
from det_models.coco_utils import get_coco_api_from_dataset
import mlflow
import time
# mlflow.pytorch.autolog(exclusive=True)
# from det_models.coco_eval import evaluate

cpu_device = torch.device('cpu')

class POIDetection(pl.LightningModule):
    def __init__(self,
                 n_classes, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model, _ = initialize_model(kwargs["backbone"], 
                                         n_classes, 
                                         tune_only=kwargs["tune_fc_only"])
        self.lr = kwargs["lr"]
        optimizers = {'adamw': optim.AdamW, 'sgd': optim.SGD}
        self.optimizer = optimizers[kwargs["optimizer"]]
        if kwargs["optimizer"] == "adam" or kwargs["optimizer"] == "adamw":
            self.opt_params = {
                "betas": (0.9, 0.999),
                "eps": kwargs["eps"],
                "weight_decay": kwargs["weight_decay"]
            }
        elif kwargs["optimizer"] == "sgd":
            self.opt_params = {
                "momentum": kwargs["momentum"],
                "weight_decay": kwargs["weight_decay"],
                "nesterov": kwargs["nesterov"]
            }
        else:
            raise ValueError("Optimizer {0} is not supported".format(kwargs["optimizer"]))

    
    # def setup_evaluator(self, coco_evaluator):
    #     self.coco_evaluator = coco_evaluator
    
    def setup_evaluator(self, eval_dataset):
        self.eval_dataset = eval_dataset
        coco = get_coco_api_from_dataset(self.eval_dataset)
        iou_types = ["bbox","segm"]
        coco_evaluator = CocoEvaluator(coco, iou_types)
        self.coco_evaluator = coco_evaluator
        
    def forward(self, images, targets=None):
        start_time_1 = time.time()
        images = list(image for image in images)
        print("--len images",len(images))
        if targets is not None :
            targets = [{k: v for k, v in t.items()} for t in targets]
            start_time_model_forward = time.time()
            outputs = self.model(images, targets)
            print("---time predict forward : ",(time.time() - start_time_model_forward))
        else:
            outputs = self.model(images)
        print("---forward : ", (time.time() - start_time_1))
        return outputs
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), 
                                   lr=self.lr, 
                                   **self.opt_params)
        return optimizer 

    def training_step(self, train_batch, batch_idx):
        start_time_2 = time.time()
        inputs, targets = train_batch
        print("--len inputs",len(inputs))
        start_time_model_predict = time.time()
        loss_dict = self(inputs, targets)
        print("---time predict training : ",(time.time() - start_time_model_predict))
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        self.log('train_loss', loss_value, prog_bar=True, logger=True)
        print("---training_step : ", (time.time() - start_time_2))
        return losses
    
    def validation_step(self, val_batch, batch_idx):
        start_time_3 = time.time()
        inputs, targets = val_batch
        
        predictions = self(inputs)
        # print("\npredictions : ", predictions)
        res = {target["image_id"].item(): output for target, output in zip(targets, predictions)}
        self.coco_evaluator.update(res)

        # self.coco_evaluator.synchronize_between_processes()
        # # accumulate predictions from all images
        # self.coco_evaluator.accumulate()
        # self.coco_evaluator.summarize()
        print("---validation_step : ",(time.time() - start_time_3))


    def validation_epoch_end(self, outputs):
        start_time_4 = time.time()
        # pass
        self.coco_evaluator.synchronize_between_processes()
        # accumulate predictions from all images
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        stats_segm = self.coco_evaluator.stats_segm
        stats_bbox = self.coco_evaluator.stats_bbox
        # with mlflow.start_run():
        self.log("val_mAP_bbox", stats_bbox[0])
        self.log("val_mAP_segm", stats_segm[0])
        coco = get_coco_api_from_dataset(self.eval_dataset)
        iou_types = ["bbox","segm"]
        coco_evaluator = CocoEvaluator(coco, iou_types)
        self.coco_evaluator = coco_evaluator
        # metrics_dict = self.coco_evaluator.summarize()
        # self.log(metrics_dict, prog_bar=True, logger=True)
        print("---validation_end : ",(time.time() - start_time_4))