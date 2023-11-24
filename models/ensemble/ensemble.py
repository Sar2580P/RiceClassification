import sys
sys.path.append('Preprocessing')
sys.path.append('models')
from utils import *
from train_eval import *
from model import *
from modules import *
from data_loading import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os 
#_______________________________________________________________________________________________________________________

hsi_obj = HSIModel(load_config('models/hsi/xception/config.yaml'))
resnet_obj = Resnet(load_config('models/rgb/resnet/config.yaml'))
enet_obj = EffecientNet(load_config('models/rgb/enet/config.yaml'))
gnet_obj = GoogleNet(load_config('models/rgb/google_net/config.yaml'))


hsi_ckpt = os.path.join('models/hsi/xception/ckpts' , os.listdir('models/hsi/xception/ckpts')[-1])
resnet_ckpt = os.path.join('models/rgb/resnet/ckpts', os.listdir('models/rgb/resnet/ckpts')[-1])
enet_ckpt = os.path.join('models/rgb/enet/ckpts/' , os.listdir('models/rgb/enet/ckpts')[-1])
gnet_ckpt = os.path.join('models/rgb/google_net/ckpts' , os.listdir('models/rgb/google_net/ckpts')[-1])

hsi_classifier = Classifier.load_from_checkpoint(hsi_ckpt, model_obj=hsi_obj)
resnet_classifier = Classifier.load_from_checkpoint(resnet_ckpt, model_obj=resnet_obj)
enet_classifier = Classifier.load_from_checkpoint(enet_ckpt, model_obj=enet_obj)
gnet_classifier = Classifier.load_from_checkpoint(gnet_ckpt, model_obj=gnet_obj)


hsi_pretrained = nn.Sequential(*list(hsi_classifier.model.children())[:-1])
resnet_pretrained = nn.Sequential(*list(resnet_classifier.model.children())[:-1])
enet_pretrained = nn.Sequential(*list(enet_classifier.model.children())[:-1])
gnet_pretrained = nn.Sequential(*list(gnet_classifier.model.children())[:-1])

for param in hsi_pretrained.parameters():
  param.requires_grad = False
  
for param in resnet_pretrained.parameters():
  param.requires_grad = False

for param in enet_pretrained.parameters():
  param.requires_grad = False

for param in gnet_pretrained.parameters():
  param.requires_grad = False

#_______________________________________________________________________________________________________________________
class Ensemble(nn.Module):
  def __init__(self , config):
    super(Ensemble, self).__init__()
    self.config = config

    self.model = nn.Sequential(
                    Dense(drop = 0.25 , in_size = 1536, out_size = 1024, ), 
                    Dense(drop = 0.15 , in_size = 1024, out_size = 256, ), 
                    Dense(drop = 0 , in_size = 256, out_size = self.config['num_classes']),
                    )

    self.layer_lr = self.model.parameters() 
                     

  def forward(self, x):
    x_hsi , x_rgb = x[0] , x[1]
    hsi_out = hsi_pretrained(x_hsi)
    resnet_out = resnet_pretrained(x_rgb)
    enet_out = enet_pretrained(x_rgb)
    gnet_out =  gnet_pretrained(x_rgb)
    out = self.model(torch.cat((hsi_out, resnet_out, enet_out, gnet_out), dim=1))
    return out

config = load_config('models/ensemble/config.yaml')
model_obj = Ensemble(config)
torch.set_float32_matmul_precision('high')
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
                  accelerator = 'gpu' ,max_epochs=100, logger=[wandb_logger,csv_logger])  
 
trainer.fit(model, tr_loader, val_loader)
trainer.test(model, tst_loader)