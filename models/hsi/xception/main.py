import wandb
import os , sys
sys.path.append(os.getcwd())
from Preprocessing.utils import load_config

def train():
  from xception import  tr_loader, val_loader, tst_loader
  from model import HSIModel
  from train_eval import Classifier
  from pytorch_lightning import Trainer
  from callbacks import early_stop_callback, rich_progress_bar, rich_model_summary
  import torch
  
  torch.set_float32_matmul_precision('high')

  config_defaults = load_config('models/hsi/xception/config.yaml')
  run_name = f"lr_{config_defaults['lr']}-bs{config_defaults['BATCH_SIZE']}-decay_{config_defaults['weight_decay']}"


  wandb.init(config=config_defaults,  name=run_name,resume=False)
  model_obj = HSIModel(config_defaults)
  model = Classifier(model_obj)

  trainer = Trainer(callbacks=[ early_stop_callback ,rich_progress_bar, rich_model_summary], 
                  accelerator = 'gpu' ,max_epochs=150) 
  
  trainer.fit(model, tr_loader, val_loader)
  trainer.test(model, tst_loader)



sweep_config = load_config('models/hsi/xception/sweep.yaml')
sweep_id = wandb.sweep(sweep_config , project='hsi_xception')
wandb.agent(sweep_id,function=train , project='hsi_xception' ,count=50)