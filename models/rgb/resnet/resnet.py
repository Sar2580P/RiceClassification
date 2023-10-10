import  sys
import pandas as pd
sys.path.append('Preprocessing')
sys.path.append('models')
from train_eval import *
from model import Resnet
from utils import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms


config_path = 'models/rgb/resnet/config.yaml'
config = load_config(config_path)

model_obj = Resnet(config)

transforms_ = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(p=0.32), 
                        transforms.RandomRotation(5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          ])

df_tr = pd.read_csv(config['tr_path']).iloc[:,[0,2]]
df_tr.columns = ['img_path' , 'class_id']
tr_dataset = MyDataset(df_tr, transforms_)
tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=4)

df_val = pd.read_csv(config['val_path']).iloc[:,[0,2]]
df_val.columns = ['img_path' , 'class_id']
val_dataset = MyDataset(df_val,transforms_)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

df_tst = pd.read_csv(config['tst_path']).iloc[:,[0,2]]
df_tst.columns = ['img_path' , 'class_id']
tst_dataset = MyDataset(df_tst, transforms_)
tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

#___________________________________________________________________________________________________________________
model = Classifier(model_obj)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=5,
   verbose=True,
   mode='min'
)


trainer = Trainer(callbacks=[early_stop_callback], max_epochs=100)
trainer.fit(model, tr_loader, val_loader)