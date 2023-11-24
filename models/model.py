from sympy import E
import torchvision
import torch.nn as nn 
from modules import *
import torch

class Resnet():
  # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html
  def __init__(self, config ):
    self.config = config
    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 20}]
    


  def get_model(self):
    self.resnet = torchvision.models.resnet50(weights = 'ResNet50_Weights.DEFAULT', progress = True)
    
    self.base_model = nn.Sequential(*list(self.resnet.children())[:-1])
    # print(self.base_model)

    self.head = nn.Sequential(
                Dense(0.2 , 2048 ,1024), 
                Dense(0.14 ,1024, 256), 
                        )
    return nn.Sequential(
                self.base_model ,
                nn.Flatten(), 
                self.head ,
                Dense(0, 256, self.config['num_classes']) ,
                        )
    
  def forward(self, x):
    return self.model(x)
#___________________________________________________________________________________________________________________  

class EffecientNet():
  # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
  def __init__(self, config):
    self.config = config
    
    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 100}]

  def get_model(self): 
    self.head = nn.Sequential(
                nn.Flatten(1) ,
                Dense(0.2 , 1280 ,512),   
              )

    self.enet = torchvision.models.efficientnet_v2_l( weights='DEFAULT' , progress = True)    # 'DEFAULT'  : 'IMAGENET1K_V1'
    self.base_model = nn.Sequential(*list(self.enet.children())[:-1])

    return nn.Sequential(
                  self.base_model ,
                  self.head, 
                  Dense(0, 512, self.config['num_classes'])
                        )   
    
  def forward(self, x):
    return self.model(x)
#___________________________________________________________________________________________________________________

class GoogleNet():
  # https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
  def __init__(self, config):
    self.config = config
    

    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 100}]

  def get_model(self): 
    self.head = nn.Sequential(
                nn.Flatten(1) ,
                Dense(0.2 , 1024 ,512), 
                
    )
    self.gnet = torchvision.models.googlenet( weights='DEFAULT' , progress = True)    # 'DEFAULT'  : 'IMAGENET1K_V1'
    self.base_model = nn.Sequential(*list(self.gnet.children())[:-2])
    return nn.Sequential(
                  self.base_model ,
                  self.head, 
                  Dense(0, 512, self.config['num_classes'])
                        )   
    
  def forward(self, x):
    return self.model(x)  
  
#___________________________________________________________________________________________________________________

class HSIModel(nn.Module):
  def __init__(self , config):
    super(HSIModel, self).__init__()
    self.config = config
    self.in_channels = self.config['in_channels']

    self.squeeze_channels = 168
    self.model = self.get_model()
    self.layer_lr = [{'params' : self.base_model.parameters()},{'params': self.head.parameters(), 'lr': self.config['lr'] * 1}]


  def get_model(self):
    self.head = nn.Sequential(
                  nn.Flatten(1) ,
                  Dense(0.25 , 1728, 512), 
                  Dense(0.15, 512, 256), 
    )
    self.base_model = nn.Sequential(
        # BandAttentionBlock(self.in_channels), 
        # SqueezeBlock(self.in_channels, self.squeeze_channels),
        XceptionBlock(self.squeeze_channels, 128), 
        XceptionBlock(128, 256), 
        ResidualBlock(256, 8),
        # XceptionBlock(128, 256), 
        SeparableConvBlock(256, 784), 
        SeparableConvBlock(784, 1728), 
        nn.MaxPool2d(kernel_size = (3,3) ,stride = (2,2)) , 
        nn.AdaptiveAvgPool2d((1,1)) ,
    )
    return nn.Sequential(
                  self.base_model,
                  self.head, 
                  Dense(0, 256, self.config['num_classes']),
                        )
    
  def forward(self, x):
  
    return self.model(x)
#___________________________________________________________________________________________________________________

# model = HSIModel({'lr': 0.001, 'num_classes': 107, 'in_channels': 168})
# print(model.base_model)
# write to txt file
# with open('models/hsi/model.txt', 'w') as f:
#     print(model, file=f)
    
# model = EffecientNet({'lr': 0.001, 'num_classes': 107})
# print(model.base_model)
# x = torch.randn(32,168, 126, 240)
# y = model.forward(x)

# m = nn.AdaptiveAvgPool2d(1)
# input = torch.randn(32, 64, 8, 9)
# output = m(input)
# print(output.shape)

# model = GoogleNet({'lr': 0.001, 'num_classes': 107})
# print(model.base_model)
# x = torch.randn(32,3, 126, 240)
# y = model.forward(x)
# print(y.shape)

# model = Resnet({'lr': 0.001, 'num_classes': 107})
# print(model.base_model)