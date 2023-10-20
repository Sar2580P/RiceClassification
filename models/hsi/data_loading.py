import  sys
import pandas as pd
sys.path.append('Preprocessing')
from train_eval import *
from callbacks import *

class_ct = 107
config = {
  'tr_path' : 'Data/{class_ct}/df_tr.csv'.format(class_ct=class_ct),
  'val_path' : 'Data/{class_ct}/df_val.csv'.format(class_ct=class_ct),
  'tst_path' : 'Data/{class_ct}/df_tst.csv'.format(class_ct=class_ct),
}

df_tr = pd.read_csv(config['tr_path']).iloc[:,[1,2]]
df_tr.columns = ['img_path' , 'class_id']
tr_dataset = MyDataset(df_tr, hsi_img_transforms)

df_val = pd.read_csv(config['val_path']).iloc[:,[1,2]]
df_val.columns = ['img_path' , 'class_id']
val_dataset = MyDataset(df_val, hsi_img_transforms)

df_tst = pd.read_csv(config['tst_path']).iloc[:,[1,2]]
df_tst.columns = ['img_path' , 'class_id']
tst_dataset = MyDataset(df_tst, hsi_img_transforms)
