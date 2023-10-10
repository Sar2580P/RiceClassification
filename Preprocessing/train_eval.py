import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchmetrics

class Classifier(pl.LightningModule):
  def __init__(self, model_obj, train_df:pd.DataFrame, val_df:pd.DataFrame, tst_df:pd.DataFrame):
    super().__init__()
    self.model = model_obj.model
    self.config = model_obj.config
    self.transforms_ = model_obj.transforms_
    self.train_df = train_df
    self.val_df = val_df
    self.tst_df = tst_df

    self.accuracy = torchmetrics.Accuracy()
    self.criterion = torch.nn.CrossEntropyLoss()

  def setup(self, stage: str):
    if stage=='fit':
      self.train_dataset = MyDataset(self.train_df, self.transforms_)
      self.val_dataset = MyDataset(self.val_df, self.transforms_)
    elif stage == 'test':
      self.tst_dataset = MyDataset(self.tst_df, self.transforms_)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size= self.config['BATCH_SIZE'], shuffle=True, num_workers=4)
  
  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=True, num_workers=4 )
  
  def test_dataloader(self):
    return DataLoader(self.tst_dataset, self.config['BATCH_SIZE'], shuffle=False, num_workers=4)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = self.criterion(y_hat, y)
    self.accuracy(y_hat, y)
    self.log("train_acc", self.accuracy, on_epoch=True,prog_bar=True, logger=True)
    self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    return loss
  
  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.model(x)
    loss = self.criterion(y_hat, y)
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
  
            
