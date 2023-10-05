import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
import os
from PIL import Image


class Classifier(pl.LightningModule):

  def __init__(self, model_obj  ,
                optimizer, 
               df_tr, df_val , lr_scheduler = None):
    
    super(Classifier, self).__init__()
    self.df_tr = df_tr
    self.df_val = df_val
    self.model_obj = model_obj
    self.optimizer = optimizer
    self.criterion = torch.nn.CrossEntropyLoss()
    self.lr_scheduler = lr_scheduler
    self.valid_loss_min = np.Inf

  def forward(self, x):
      return self.model_obj.forward(x)

  def loss_function(self, logits, labels):
    return self.loss(logits, labels)

  def __training_step__(self)->dict:
      self.model_obj.model.train()
      epoch_acc, epoch_loss = 0, 0
      pbar = tqdm(self.tr_loader)

      for batch_idx, (inputs, targets) in enumerate(pbar):
          # inputs, targets = inputs.to(self.device), targets.to(self.device)
          outputs = self.forward(inputs)
          loss = self.criterion(outputs, targets.long())
          self.optimizer.zero_grad()    # clear the gradients of all optimized variables
          loss.backward()               # back_prop: compute gradient of the loss with respect to model parameters
          self.optimizer.step()         # parameter update
          epoch_loss += loss.item()
          _, pred = torch.max(outputs, dim=1)
          correct = torch.sum(pred==targets).item()
          total = targets.size(0)
          epoch_acc += correct/total

      tr_loss = epoch_loss/self.total_steps_tr
      tr_acc = epoch_acc/self.total_steps_tr
      return {'loss':tr_loss, 'acc':tr_acc}
  

  def __validation_step__(self)->dict:
      self.model_obj.model.eval()
      epoch_loss, epoch_acc = 0 , 0
      pbar = tqdm(self.val_loader)

      with torch.no_grad():           # disable gradient calculation
        for batch_idx, (inputs, targets) in enumerate(pbar):
          # inputs, targets = inputs.to(self.device), targets.to(self.device)
          outputs = self.forward(inputs)
          loss = self.criterion(outputs, targets.long())

          epoch_loss += loss.item()
          _, pred = torch.max(outputs, dim=1)
          correct = torch.sum(pred==targets).item()
          total = targets.size(0)
          epoch_acc += correct/total

        val_loss = epoch_loss/self.total_steps_val
        val_acc = epoch_acc/self.total_steps_val
      return {'loss':val_loss, 'acc':val_acc}
  
  def __prepare_data__(self):
    # transforms for images
    tr_dataset = MyDataset(self.df_tr , self.model_obj.transforms_)
    self.tr_loader = DataLoader(tr_dataset, batch_size=self.model_obj.config['BATCH_SIZE'], shuffle=True)
    val_dataset = MyDataset(self.df_val , self.model_obj.transforms_)
    self.val_loader = DataLoader(val_dataset, batch_size=self.model_obj.config['BATCH_SIZE'], shuffle=True)
    self.total_steps_tr = len(self.tr_loader)
    self.total_steps_val = len(self.val_loader)
  
  def run(self, epochs):
    self.__prepare_data__()

    train_losses, train_acc = [] , []
    test_losses, test_acc  = [] , []
    
    for epoch in range(epochs):
      print(f'\nEpoch: {epoch}')
      train_logs = self.__training_step__()
      val_logs = self.__validation_step__()
      
      
      
      print ('Epoch [{}] --> LossTr: {:.4f}    AccTr: {:.4f}'
              .format(epoch, train_logs['loss'],train_logs['acc']), end = '    ')
      print('lossVal : {:.4f}     accVal : {:.4f}\n'.format(val_logs['loss'] , val_logs['acc']))

      train_losses.append(train_logs['loss'])
      train_acc.append(train_logs['acc'])
      test_losses.append(val_logs['loss'])
      test_acc.append(val_logs['acc'])

      if self.lr_scheduler is not None:
        self.lr_scheduler.step(train_losses[-1])   # updating learning rate

      network_learned = self.valid_loss_min - test_losses[-1] > 0.01
      if network_learned:
        self.valid_loss_min = test_losses[-1]
        torch.save(self.model_obj.model.state_dict(), os.path.join(self.model_obj.config['dir'] , 'model_{name}.pt'.format(name = self.model_obj.config['name'])))
        print('Detected network improvement, saving current model  "\u2705"')

    history = {
      'train_loss': train_losses,
      'train_acc': train_acc,
      'val_loss': test_losses,
      'val_acc': test_acc
    }

    with open(os.path.join(self.model_obj.dir ,'history.pickle'), 'wb') as handle:
      pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
      handle.close()

    return history
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
  
            
