import torchvision
import torch.nn as nn 
import sys
sys.path.append('enet')

class Resnet():
  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html
  def __init__(self, config ):
    self.config = config
    self.resnet = torchvision.models.resnet101(weights = 'DEFAULT', progress = True)
    self.base_model = nn.Sequential(*list(self.resnet.children())[:-1])

    self.__create_model__()
    self.layer_lr = [{'params' : self.base_model},{'params': self.head, 'lr': self.config['lr'] * 100}]

  def __create_model__(self):
    self.head = nn.Sequential(
                Dense(0.2 , 2048 ,1024), 
                Dense(0.15, 1024, 512) ,
                Dense(0, 512, self.config['num_classes'])
    )
    self.model = nn.Sequential(
                self.base_model ,
                self.head
                        )
    
  def forward(self, x):
    x =  self.model(x) 
    return x
#___________________________________________________________________________________________________________________  

class EffecientNet():
  # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
  def __init__(self, config):
    self.config = config
    self.enet = torchvision.models.efficientnet_v2_l( weights='DEFAULT' , progress = True)    # 'DEFAULT'  : 'IMAGENET1K_V1'
    self.base_model = nn.Sequential(*list(self.enet.children())[:-1])

    self.__create_model__()
    self.layer_lr = [{'params' : self.base_model},{'params': self.head, 'lr': self.config['lr'] * 100}]

  def __create_model__(self): 
    self.head = nn.Sequential(
                Dense(0.2 , 1028 ,512), 
                Dense(0, 512, self.config['num_classes'])
    )
    self.model = nn.Sequential(
                  self.base_model ,
                  self.head
                        )  
    return 
    
  def forward(self, x):
    x =  self.model(x) 
    return x



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
