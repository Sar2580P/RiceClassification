import torchvision
import torch.nn as nn 
from modules import *
import torch

class Resnet():
  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html
  def __init__(self, config ):
    self.config = config
    self.resnet = torchvision.models.resnet101(weights = 'ResNet101_Weights.DEFAULT', progress = True)
    self.base_model = nn.Sequential(*list(self.resnet.children())[:-1])

    self.__create_model__()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 100}]

  def __create_model__(self):
    self.head = nn.Sequential(
                Dense(0.2 , 2048 ,1024), 
                Dense(0.15, 1024, 512) ,
                Dense(0, 512, self.config['num_classes'])
    )
    self.model = nn.Sequential(
                self.base_model ,
                nn.Flatten(), 
                self.head ,
                        )
    
  def forward(self, x):
    return self.model(x)
#___________________________________________________________________________________________________________________  

class EffecientNet():
  # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
  def __init__(self, config):
    self.config = config
    self.enet = torchvision.models.efficientnet_v2_l( weights='DEFAULT' , progress = True)    # 'DEFAULT'  : 'IMAGENET1K_V1'
    self.base_model = nn.Sequential(*list(self.enet.children())[:-1])

    self.__create_model__()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 100}]

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
 

class HSIModel(nn.Module):
  def __init__(self , config):
    super(HSIModel, self).__init__()
    self.config = config
    self.in_channels = self.config['in_channels']
    self.head = nn.Sequential(Dense(0.2 , 1024, 256), 
                              Dense(0, 256, self.config['num_classes'])
                )
    self.base_model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 1}]


  def get_model(self):
    return nn.Sequential(
        BandAttentionBlock(self.in_channels), 
        SqueezeBlock(self.in_channels, self.squeeze_channels),
        XceptionBlock(self.squeeze_channels, 128), 
        XceptionBlock(128, 256) , 
        ResidualBlock(256, 256) ,
        XceptionBlock(256, 512), 
        SeparableConvBlock(512, 1024), 
        nn.MaxPool2d(kernel_size = (3,3) , stride = (2,2)) , 
        nn.AdaptiveAvgPool2d((1,1)) ,
    )
  def forward(self, x):
    x = self.base_model(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x
#___________________________________________________________________________________________________________________

# model = HSIModel(168, 107)

# x = torch.randn(32,168, 256, 256)
# y = model(x)
    
