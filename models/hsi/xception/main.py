from json import load
from traitlets import default
import wandb
import os , sys
sys.path.append(os.getcwd())
from Preprocessing.utils import load_config

def train():
  from xception import trainer, model ,tr_loader, val_loader, tst_loader
  from model import HSIModel
  from train_eval import Classifier
  from pytorch_lightning import Trainer
  from pytorch_lightning.loggers import WandbLogger, CSVLogger
  from callbacks import early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary

  config_defaults = load_config('models/hsi/xception/config.yaml')
  wandb.init(config=config_defaults)
  model_obj = HSIModel(config_defaults)
  model = Classifier(model_obj)
  wandb_logger = WandbLogger(project=config_defaults['model_name'])
  csv_logger = CSVLogger(config_defaults['dir'], name=config_defaults['model_name']+'_logs')

  trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary], 
                  accelerator = 'cpu' ,max_epochs=1, logger=[wandb_logger,csv_logger]) 
  
  trainer.fit(model, tr_loader, val_loader)
  trainer.test(model, tst_loader)



sweep_config = load_config('models/hsi/xception/sweep.yaml')
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id,function=train)
