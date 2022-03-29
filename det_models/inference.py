import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import cv2
import os
import csv
from PIL import Image,ImageDraw
from datasets.dataset import Labelizer

def _read_csv(filepath):
    images = []
    # reading csv file
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for img in csvreader:
            img = img[0]
            images.append(img)
            # labels.append([label])
    return images

class Color_convert():
    def __init__(self):
        super().__init__()
        
        self.labels = {'retail': "red", 'house': "yellow", 'office': "blue", 'building': "white", 'site': "orange", 'gate': "brown"}
        
    def transform(self, label):
        return self.labels[label]
    
    def num_classes(self):
        return len(self.labels)

class POIDetectionTask(pl.LightningModule):
    def __init__(self,
                 model,
                 output_path,
                 test_path,
                 data_path):
        super().__init__()
        
        self.model = model
        self.output_path = output_path
        self.test_path = test_path
        self.data_path = data_path
        self.pred_targets = []
        self.true_targets = []
        self.pred_labels = []
        self.img_dir = []
        self.labelizer = Labelizer()
        self.color_convert = Color_convert()

    def forward(self, x):
        output = self.model(x)
        return output

    def predict_step(self, test_batch, batch_idx):
        images, targets = test_batch

        #
        outputs = self(images)
        for target in outputs:
            shape = target['boxes']
            score = target['scores']
            labels = target['labels']
            shape = shape.numpy()
            score = score.numpy()
            labels = labels.numpy()
            select_shape = []
            select_labels = []
            for i in range(0,len(score)):
                if (score[i]>0.5):
                    select_shape.append(shape[i])
                    select_labels.append(labels[i])
            select_shape = [select_shape]
            select_labels = [select_labels]
            self.pred_targets.extend(select_shape)
            self.pred_labels.extend(select_labels)

        #
        for target in targets:
            shape = target['boxes']
            shape = shape.numpy()
            shape = [shape]
            self.true_targets.extend(shape)
    
    def on_predict_end(self):
        # pass
        list_img = _read_csv(self.test_path)
        for i in range(0,len(list_img)):
            img_path = os.path.join(self.data_path,list_img[i])
            image = Image.open(img_path).convert('RGB')
            pred_boxes = self.pred_targets[i]
            pred_labels = self.pred_labels[i]
            true_boxes = self.true_targets[i]
            draw = ImageDraw.Draw(image)
            for box in true_boxes:
                draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline = "green")
            for j in range(0,len(pred_boxes)):
                label = pred_labels[j]
                label = self.labelizer.inverse_transform(label)
                box = pred_boxes[j]
                draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline = self.color_convert.transform(label))
            link = os.path.join(self.output_path,list_img[i])
            image.save(link, format="png")