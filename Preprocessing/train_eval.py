import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchmetrics

class Classifier(pl.LightningModule):
  def __init__(self, model_obj):
    super().__init__()
    self.model = model_obj.model
    self.config = model_obj.config

    self.accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['num_classes'])
    self.criterion = torch.nn.CrossEntropyLoss()

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = self.criterion(y_hat, y.long())
    self.accuracy(y_hat, y)
    self.log("train_acc", self.accuracy, on_epoch=True,prog_bar=True, logger=True)
    self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = self.criterion(y_hat, y.long())
    self.accuracy(y_hat, y)
    self.log("val_acc", self.accuracy, on_epoch=True,prog_bar=True, logger=True)
    self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = self.criterion(y_hat, y)
    self.accuracy(y_hat, y)
    self.log("test_acc", self.accuracy, on_epoch=True,prog_bar=True, logger=True)
    self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def configure_optimizers(self):
    optim =  torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, threshold=0.001, cooldown =2,verbose=True)
    return [optim], [{'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'train_loss', 'name': 'lr_scheduler'}]

#___________________________________________________________________________________________________________________
class MyDataset(Dataset):
  # defining values in the constructor
  def __init__(self , df,transforms):
    self.df = df
    self.Y = torch.tensor( self.df.iloc[:, -1].values, dtype=torch.float32)
    self.shape = self.df.shape
    self.transforms = transforms
    
  # Getting the data samples
  def __getitem__(self, idx):
    y =  self.Y[idx]
    img_tensor = None
    img_path = self.df.img_path.iloc[idx]
    if img_path.lower().split('.')[-1] == 'jpg':
      img_tensor = Image.open(img_path)
    else :
      img_tensor = np.load(img_path)

    img_tensor = self.transforms(img_tensor)
    return img_tensor, y
  
  def __len__(self):
    return self.shape[0]
  
            
