import os, sys
import pandas as pd
import torch
sys.path.append('Preprocessing')
sys.path.append('models')
from train_eval import *
from model import Resnet
from utils import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
config_path = 'models/rgb/resnet/config.yaml'
config = load_config(config_path)

model_obj = Resnet(config)
optimizer = torch.optim.Adam(model_obj.model.parameters(), lr=config['lr'])

df_tr = pd.read_csv(config['tr_path']).iloc[:,[0,2]]
df_tr.columns = ['img_path' , 'class_id']

df_val = pd.read_csv(config['val_path']).iloc[:,[0,2]]
df_val.columns = ['img_path' , 'class_id']

clf = Classifier(model_obj, optimizer, df_tr, df_val)
clf.run(50)