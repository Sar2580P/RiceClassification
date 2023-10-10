import torchvision
import torch.nn as nn 
import sys
sys.path.append('enet')

class Resnet():
  def __init__(self, config ):
    print('in resnet constructor')
    self.config = config
    self.model = torchvision.models.resnet101(pretrained = True, progress = True)
    
    self.__create_model__()
    # print(summary(self.model, (3, 224, 224)))

  def __create_model__(self):
    for param in self.model.parameters():
      param.requires_grad = True 

    self.model.fc = nn.Sequential(
                Dense(0.2 , 2048 ,1024), 
                Dense(0.15, 1024, 512) ,
                Dense(0, 512, self.config['num_classes'])
                        )
    return self.model
    
  def forward(self, x):
    x =  self.model(x) 
    return x
#___________________________________________________________________________________________________________________  
'''
class EffecientNet():
  def __init__(self, config):
    self.config = config
    from efficientnet_pytorch import EfficientNet
    model_name = 'efficientnet-b7'
    self.enet  = EfficientNet.from_pretrained(model_name, num_classes=107).to(device)

    print(summary(self.enet, (3, 224, 224)))

    self.model = self.__create_model__()

  def __create_model__(self):
    for param in self.enet.parameters():
      param.requires_grad = False 

    self.enet.fc = nn.Sequential(
                Dense(0.2 , 2048 ,1024), 
                Dense(0.15, 1024, 512) ,
                Dense(0, 512, self.config['num_classes']) 
                        )
    return self.enet
    
  def forward(self, x):
    x =  self.resnet(x) 
    return x
'''
#___________________________________________________________________________________________________________________
class Dense(nn.Module):
    def __init__(self, drop ,in_size, out_size):
        super(Dense ,self).__init__()
        self.dropout = nn.Dropout(drop)
        self.linear = nn.Linear(in_size, out_size)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.prelu(x)
        return x
# sys.path.append('Preprocessing')

# from utils import *

# config_path = 'models/rgb/enet/config.yaml'
# config = load_config(config_path)
# x = EffecientNet(config)