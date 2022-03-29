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
from det_models.check_pass_box import check_box
# from datasets.dataset import Labelizer
from datasets_detection.dataset import Labelizer
from matplotlib import cm
import random

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
        
        # self.labels = {'ten bien': "red", 'ten duong': "yellow"}
        self.labels = {'dem': "red"}
        
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
        self.pred_masks = []
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
            masks = target['masks']
            shape = shape.numpy()
            masks = masks.numpy()
            score = score.numpy()
            labels = labels.numpy()
            select_shape = []
            select_masks = []
            select_labels = []
            for i in range(0,len(score)):
                if (score[i]>0.9):
                    select_shape.append(shape[i])
                    select_masks.append(masks[i])
                    select_labels.append(labels[i])
            select_shape = [select_shape]
            select_masks = [select_masks]
            select_labels = [select_labels]
            self.pred_targets.extend(select_shape)
            self.pred_masks.extend(select_masks)
            self.pred_labels.extend(select_labels)

        #
        # for target in targets:
        #     shape = target['boxes']
        #     # shape = target['masks']
        #     shape = shape.numpy()
        #     shape = [shape]
        #     self.true_targets.extend(shape)
    
    def on_predict_end(self):
        # pass
        list_img = _read_csv(self.test_path)
        num_true_boxes = 0
        num_pred_true_boxes = 0
        num_pred_boxes = 0
    
        for i in range(0,len(list_img)):
            img_path = os.path.join(self.data_path,list_img[i])
            image = Image.open(img_path).convert('RGBA')
            pred_boxes = self.pred_targets[i]
            pred_masks = self.pred_masks[i]
            pred_labels = self.pred_labels[i]
            

            # true_boxes = self.true_targets[i]
            # draw = ImageDraw.Draw(image)
            final = Image.new("RGBA", image.size)
            final.paste(image, (0,0), image)
            draw = ImageDraw.Draw(final)
            # num_true_boxes += len(true_boxes)
            # num_pred_boxes += len(pred_boxes)
            # for box in true_boxes:
            #     draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline = "green")
            #     for pred_box in pred_boxes:
            #         if check_box(box,pred_box):
            #             num_pred_true_boxes +=1
            width, height = image.size
            binary_mask = np.zeros((height,width),dtype="uint8")
            for j in range(0,len(pred_boxes)):
                label = pred_labels[j]
                label = self.labelizer.inverse_transform(label)
                box = pred_boxes[j]
                # mask = pred_masks[j]
                # print(type(mask[0]))
                # binary_mask[:,:] = binary_mask[:,:] + mask[0]
                # #create new layer mask
                
                # colImage = np.zeros((height,width,3), dtype="uint8")
                # colImage[:,:,0] = mask[0]*random.randrange(0,255,3)
                # colImage[:,:,1] = mask[0]*random.randrange(0,255,3)
                # colImage[:,:,2] = mask[0]*random.randrange(0,255,3)
                # img = Image.fromarray(np.uint8(colImage))
                # img = img.convert("RGBA")
                
                # new_img = []
                # datas = img.getdata()
                # for item in datas:
                #     if (item[0] < 50) and (item[1] < 50) and (item[2] < 50):
                #         new_img.append((255, 255, 255, 0))
                #     else:
                #         new_img.append(item)
                # img.putdata(new_img)
                # img = img.convert("RGBA")

                # final.paste(img, (0,0), img)

                #draw box of signboard
                draw.rectangle(((box[0], box[1]), (box[2], box[3])),width=10, outline = self.color_convert.transform(label))

            link = os.path.join(self.output_path,list_img[i])
            final.save(link, format="png")