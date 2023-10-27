from json import load
import sys
sys.path.append('Preprocessing')
from utils import *
from train_eval import *
from model import *


hsi_ckpt = 'models/hsi/xception/ckpts/epoch=0-step=0.ckpt'
resnet_ckpt = 'models/hsi/resnet/ckpts/epoch=0-step=0.ckpt'
enet_ckpt = 'models/hsi/enet/ckpts/epoch=0-step=0.ckpt'
gnet_ckpt = 'models/hsi/googlenet/ckpts/epoch=0-step=0.ckpt'



class Ensemble(nn.Module):
  def __init__(self , config):
    super(Ensemble, self).__init__()
    self.config = config
    self.hsi_model = HSIModel.load_from_checkpoint(hsi_ckpt)
    self.resnet_model = Resnet.load_from_checkpoint(resnet_ckpt)
    self.enet_model = EffecientNet.load_from_checkpoint(enet_ckpt)
    self.gnet_model = GoogleNet.load_from_checkpoint(gnet_ckpt)

    self.w_hsi = nn.Parameter(torch.tensor(0.25))
    self.w_resnet = nn.Parameter(torch.tensor(0.25))
    self.w_enet = nn.Parameter(torch.tensor(0.25))
    self.w_gnet = nn.Parameter(torch.tensor(0.25))

    self.layer_lr = [{'params' : self.hsi_model.parameters()},{'params': self.resnet_model.parameters()},{'params': self.enet_model.parameters()},{'params': self.gnet_model.parameters()}]

  def forward(self, x):
    hsi_out = self.hsi_model(x)
    resnet_out = self.resnet_model(x)
    enet_out = self.enet_model(x)
    gnet_out = self.gnet_model(x)

    out = self.w_hsi*hsi_out + self.w_resnet*resnet_out + self.w_enet*enet_out + self.w_gnet*gnet_out
    return out

config = load_config('models/ensemble/config.yaml')
model = Ensemble(config)
print(model)