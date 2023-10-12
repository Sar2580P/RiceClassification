import  sys
import pandas as pd
sys.path.append('Preprocessing')
from train_eval import *
from callbacks import *
config = {
  'tr_path' : 'Data/df_tr.csv',
  'val_path' : 'Data/df_val.csv',
  'tst_path' : 'Data/df_tst.csv',
}

df_tr = pd.read_csv(config['tr_path']).iloc[:,[0,2]]
df_tr.columns = ['img_path' , 'class_id']
tr_dataset = MyDataset(df_tr, img_transforms)

df_val = pd.read_csv(config['val_path']).iloc[:,[0,2]]
df_val.columns = ['img_path' , 'class_id']
val_dataset = MyDataset(df_val, img_transforms)

df_tst = pd.read_csv(config['tst_path']).iloc[:,[0,2]]
df_tst.columns = ['img_path' , 'class_id']
tst_dataset = MyDataset(df_tst, img_transforms)
