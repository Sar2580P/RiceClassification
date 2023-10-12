import  sys
sys.path.append('models')
sys.path.append('Preprocessing')
sys.path.append('models/rgb')
from train_eval import *
from model import Resnet
from callbacks import *
from utils import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from data_loading import *


config_path = 'models/rgb/resnet/config.yaml'
config = load_config(config_path)

model_obj = Resnet(config)

#___________________________________________________________________________________________________________________
num_workers = 0
tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)
tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)

#___________________________________________________________________________________________________________________
model = Classifier(model_obj)

checkpoint_callback.dirpath = os.path.join(config['dir'], 'ckpts')
checkpoint_callback.filename = config['ckpt_file_name']

wandb_logger = WandbLogger(project=config['model_name'])
csv_logger = CSVLogger(config['dir'], name=config['model_name']+'_logs')

trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary], 
                  accelerator = 'gpu' ,max_epochs=1, logger=[wandb_logger,csv_logger])  
 
trainer.fit(model, tr_loader, val_loader)