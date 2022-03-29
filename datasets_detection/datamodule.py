import csv
from math import degrees

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

from datasets_signboard_detection.dataset import PoIDataset
import datasets_signboard_detection.utils as utils

import time



def _read_csv(filepath):
    images, labels = [], []
    # reading csv file
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for img in csvreader:
            images.append(img)
            # labels.append([label])
    return images


class POIDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 data_path: str,
                 label_path:str,
                 train_batch_size=8,
                 test_batch_size=8,
                 seed=28):
        super().__init__()

        self.train_path = train_path
        self.test_path = test_path
        self.data_path = data_path
        self.label_path = label_path
        
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
    

    def prepare_data(self):
        pass
    
    def setup(self, stage="fit"):
        transform = [transforms.ToTensor()]
        # transform = [transforms.ToTensor()]
        train_transform = transforms.Compose(transform)
        test_transform = transforms.Compose(transform)
        
        if stage == "fit" or stage is None:
            image_names = _read_csv(self.train_path)
            x_train, x_val = train_test_split(image_names, test_size=0.3, random_state=self.seed)
            start_time_train_dataset = time.time()
            self.train_dataset = PoIDataset(x_train,
                                            self.data_path,
                                            self.label_path,
                                            transforms=train_transform)
            print("---train_dataset_create : ", (time.time() - start_time_train_dataset))
            start_time_val_dataset = time.time()
            self.val_dataset = PoIDataset(x_val,
                                          self.data_path, 
                                          self.label_path,
                                          transforms=test_transform)
            print("---val_dataset_create : ", (time.time() - start_time_val_dataset))

        if stage == "predict" or stage is None:
            image_names = _read_csv(self.test_path)
            self.test_dataset = PoIDataset(image_names,
                                           self.data_path,
                                           self.label_path,  
                                           transforms=test_transform)


    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(self.train_dataset,
                              batch_size=self.train_batch_size,
                              shuffle=True, 
                              num_workers=1,
                              collate_fn=utils.collate_fn)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=1,
                              collate_fn=utils.collate_fn
                              )

    def predict_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(self.test_dataset,
                              batch_size=self.test_batch_size,
                              shuffle=False,
                              num_workers=1,
                              collate_fn=utils.collate_fn)