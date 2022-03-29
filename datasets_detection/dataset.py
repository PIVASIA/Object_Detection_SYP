import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import numpy as np
import random
import time
# from datasets_signboard_detection import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class Labelizer():
    def __init__(self):
        super().__init__()
        
        # self.labels = {'background': 0, 'bien': 1, 'tenbien': 1, 'ten bien': 1, 'ten duong': 2, 'tenduong': 2}
        # self.inv_labels = {0: 'background', 1: 'ten bien', 2: 'ten duong'}
        self.labels = {'background': 0, 'dem': 1}
        self.inv_labels = {0: 'background', 1: 'dem'}
    def transform(self, label):
        return self.labels[label]
    
    def inverse_transform(self, ys):
        return self.inv_labels[ys]
    
    def num_classes(self):
        return len(self.labels)

class PoIDataset(Dataset):
    def __init__(self,
                 img_names, 
                 data_dir,
                 label_dir,
                 transforms=None):
        self.img_names = img_names
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.labelizer = Labelizer()
        # self.transforms = get_transform(train=True)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        basename = img_name[0]
        basename = basename[:-4]
        # basename = img_name[0].split(".")[0]
        img_path = os.path.join(self.data_dir, img_name[0])

        image = Image.open(img_path).convert('RGB')
        
        labels = []
        boxes = []
        masks = []
        target = {}
        # Read label json file
        if (self.label_dir is not None):
            label_path = os.path.join(self.label_dir, "%s.json" % basename)
            with open(label_path, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
                
                for shape in data['shapes']:
                    label = shape['label']
                    label = self.labelizer.transform(label)
                    point = shape['points']
                    num_point = len(point)
                    x_min = 720
                    y_min = 1280
                    x_max = 0
                    y_max = 0
                    for i in range(0, num_point):
                        x_min = min(point[i][0],x_min)
                        y_min = min(point[i][1],y_min)
                        x_max = max(point[i][0],x_max)
                        y_max = max(point[i][1],y_max)
                    if (x_min < x_max) and (y_min < y_max):
                        box = [x_min, y_min, x_max, y_max]
                        boxes.append(box)
                        labels.append(label)
                        # labels.append(1)

                    #create_mask
                        img = Image.new('L', (720, 1280), 0)
                        polygon = []
                        for p in point:
                            x = p[0]
                            y = p[1]
                            polygon.append(x)
                            polygon.append(y)
                        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                        mask = np.array(img)
                        masks.append(mask)
            
            # Create target
            num_objs = len(labels)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            image_id = torch.tensor([idx])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            # image, target = self.transforms(image, target)
            image = self.transforms(image)
        return image, target