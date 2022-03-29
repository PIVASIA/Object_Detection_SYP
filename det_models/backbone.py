import torch.nn as nn
import torchvision.models.detection as models
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def set_parameter_requires_grad(model, 
                                tune_only: bool = False):
    if tune_only:
        for child in list(model.children()):
            for param in child.parameters():
                param.requires_grad = False


def initialize_model(model_name: str, 
                     num_classes: int, 
                     tune_only: bool = False, 
                     use_pretrained: bool = True):
    input_size = 0

    model = getattr(models, model_name, lambda: None)
    model_ft = model(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, tune_only)

    if model_name.startswith("maskrcnn"):
        mask_predictor_in_channels = 256
        mask_dim_reduced = 256
        model_ft.mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)
        # in_channels = model_ft.head.classification_head.cls_logits.in_channels
        # num_anchors = model_ft.head.classification_head.num_anchors
        # # replace the pre-trained head with a new one
        # model_ft.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)

        # just another way to replace
        # model_ft.head.classification_head.cls_logits =\
        #     nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        # model_ft.head.classification_head.num_classes = num_classes

    elif model_name.startswith("fasterrcnn"):
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        # get number of input features for the classifier
        in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
        model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:
        raise ValueError("{0} is not supported!".format(model_name))

    return model_ft, input_size