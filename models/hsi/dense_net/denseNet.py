import  sys
sys.path.append('models')
sys.path.append('Preprocessing')
sys.path.append('models/hsi')
from train_eval import *
from model import DenseNet
from callbacks import *
from utils import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
# from data_loading import *

config_path = 'models/hsi/dense_net/config.yaml'
config = load_config(config_path)
torch.set_float32_matmul_precision('high')

model_parameters={}
model_parameters['densenet121'] = [6,12,24,16]
model_parameters['densenet169'] = [6,12,32,32]
model_parameters['densenet201'] = [6,12,48,32]
model_parameters['densenet264'] = [6,12,64,48]

model_obj = DenseNet(densenet_variant = model_parameters['densenet121'] , in_channels=152, num_classes=96 , compression_factor=0.3 , k = 32 , config=config)

# hsi_ckpt = os.path.join('models/hsi/xception/ckpts' , os.listdir('models/hsi/xception/ckpts')[-1])
# model = Classifier.load_from_checkpoint(hsi_ckpt, model_obj=model_obj)

#___________________________________________________________________________________________________________________
# num_workers = 8
# tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=num_workers)
# val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)
# tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)

# #___________________________________________________________________________________________________________________
# model = Classifier(model_obj)

# checkpoint_callback.dirpath = os.path.join(config['dir'], 'ckpts')
# checkpoint_callback.filename = config['ckpt_file_name']


# run_name = f"lr_{config['lr']} *** bs{config['BATCH_SIZE']} *** decay_{config['weight_decay']}"
# wandb_logger = WandbLogger(project=config['model_name'], name = run_name , log_model='all')
# csv_logger = CSVLogger(config['dir'], name=config['model_name']+'_logs')


# trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary], 
#                   accelerator = 'cpu' ,max_epochs=0, logger=[wandb_logger,csv_logger])  
 
# trainer.fit(model, tr_loader, val_loader)
# trainer.test(model, tst_loader)

