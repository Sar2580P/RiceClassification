import sys
sys.path.append('Preprocessing')
sys.path.append('models')
from utils import *
from train_eval import *
from model import *
from ensemble.data_loading import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger

#_______________________________________________________________________________________________________________________

hsi_obj = HSIModel(load_config('models/hsi/xception/config.yaml'))
resnet_obj = Resnet(load_config('models/hsi/resnet/config.yaml'))
enet_obj = EffecientNet(load_config('models/hsi/enet/config.yaml'))
gnet_obj = GoogleNet(load_config('models/hsi/googlenet/config.yaml'))

hsi_ckpt = 'models/hsi/xception/ckpts/xception--epoch=1-val_loss=3.27-val_accuracy=0.15.ckpt'
resnet_ckpt = 'models/hsi/resnet/ckpts/resnet--epoch=1-val_loss=3.27-val_accuracy=0.15.ckpt'
enet_ckpt = 'models/hsi/enet/ckpts/enet--epoch=1-val_loss=3.27-val_accuracy=0.15.ckpt'
gnet_ckpt = 'models/hsi/googlenet/ckpts/googlenet--epoch=1-val_loss=3.27-val_accuracy=0.15.ckpt'


hsi_classifier = Classifier.load_from_checkpoint(hsi_ckpt, model_obj=hsi_obj)
resnet_classifier = Classifier.load_from_checkpoint(resnet_ckpt, model_obj=resnet_obj)
enet_classifier = Classifier.load_from_checkpoint(enet_ckpt, model_obj=enet_obj)
gnet_classifier = Classifier.load_from_checkpoint(gnet_ckpt, model_obj=gnet_obj)

#_______________________________________________________________________________________________________________________
class Ensemble(nn.Module):
  def __init__(self , config):
    super(Ensemble, self).__init__()
    self.config = config

    self.w_hsi = nn.Parameter(torch.tensor(0.25))
    self.w_resnet = nn.Parameter(torch.tensor(0.25))
    self.w_enet = nn.Parameter(torch.tensor(0.25))
    self.w_gnet = nn.Parameter(torch.tensor(0.25))

    self.layer_lr = [{'params' : self.w_hsi},{'params': self.w_resnet},{'params': self.w_enet},{'params': self.w_gnet}]

  def forward(self, x_hsi , x_rgb):
    hsi_out = hsi_classifier(x_hsi)
    resnet_out = resnet_classifier(x_rgb)
    enet_out = enet_classifier(x_rgb)
    gnet_out = gnet_classifier(x_rgb)

    out = self.w_hsi*hsi_out + self.w_resnet*resnet_out + self.w_enet*enet_out + self.w_gnet*gnet_out
    return out
  
config = load_config('models/ensemble/config.yaml')
model_obj = Ensemble(config)
print(model_obj)

#_______________________________________________________________________________________________________________________

num_workers = 8
tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)
tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)
#_______________________________________________________________________________________________________________________

model = Classifier(model_obj= model_obj)

checkpoint_callback.dirpath = os.path.join(config['dir'], 'ckpts')
checkpoint_callback.filename = config['ckpt_file_name']

wandb_logger = WandbLogger(project=config['model_name'])
csv_logger = CSVLogger(config['dir'], name=config['model_name']+'_logs')

#_______________________________________________________________________________________________________________________
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary], 
                  accelerator = 'gpu' ,max_epochs=10, logger=[wandb_logger,csv_logger])  
 
trainer.fit(model, tr_loader, val_loader)
trainer.test(model, tst_loader)