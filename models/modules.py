
# Modules for hsi-model

from torch import Tensor
import torch.nn as nn

class Dense(nn.Module):
    def __init__(self, drop ,in_size, out_size):
      super(Dense ,self).__init__()
      self.drop , self.in_size , self.out_size = drop , in_size , out_size
      self.model = self.get_model()
    def get_model(self):
      return nn.Sequential(
          nn.Dropout(self.drop), 
          nn.Linear(self.in_size, self.out_size) , 
          nn.PReLU()
      )
    def forward(self, x):
      return self.model(x)
  
#___________________________________________________________________________________________________________________

class Conv2dBlock(nn.Module):
  def __init__(self , in_channels, out_channels, kernel_size=(3,3),padding = 'same',  stride = 1):
    super(Conv2dBlock, self).__init__()
    self.in_channels , self.out_channels = in_channels , out_channels
    self.kernel_size , self.padding , self.stride = kernel_size , padding , stride
    self.model = self.get_model()

  def get_model(self):
    return nn.Sequential(
        nn.BatchNorm2d(self.in_channels ),  
        nn.PReLU(),
        nn.Conv2d(in_channels = self.in_channels  , out_channels = self.out_channels , 
                  kernel_size = self.kernel_size ,padding = self.padding)
    )
  def forward(self, x):
    return self.model(x)
  
#___________________________________________________________________________________________________________________

class BandAttentionBlock(nn.Module):
  def __init__(self, in_channels, r=2):
      super(BandAttentionBlock ,self).__init__()
      self.conv2d_a = Conv2dBlock(in_channels = in_channels , out_channels = 16)
      self.conv2d_b = Conv2dBlock(in_channels = 16 , out_channels = 32)
      self.conv2d_c = Conv2dBlock(in_channels = 32 , out_channels = 32)
      self.conv2d_d = Conv2dBlock(in_channels = 32 , out_channels = 32)
      self.conv2d_e = Conv2dBlock(in_channels = 32 , out_channels = 32)
      self.max_pool = nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2))
      self.gap = nn.AdaptiveAvgPool2d((1,1))        
      self.conv1d_a = nn.Conv1d(in_channels = 1 , out_channels = in_channels//r , kernel_size = 32)
      self.conv1d_b = nn.Conv1d(in_channels = 1 , out_channels = in_channels , kernel_size = in_channels//r )
      self.sigmoid = nn.Sigmoid()

      self.att_model = self.get_model()

  def get_model(self):
    return nn.Sequential(
        self.conv2d_a ,
        self.max_pool, 
        self.conv2d_b ,
        self.conv2d_c ,
        self.max_pool, 
        self.conv2d_d ,
        self.conv2d_e ,
        self.gap ,
    )
  def forward(self, x):
    vector = self.att_model(x)   # (B,32,1,1)
    vector = vector.squeeze(3)  # (B,32,1)
    vector = vector.permute(0,2,1)  # (B,1,32)
    vector = self.conv1d_a(vector)  # (B, 84 , 1),  when --> r=2
    vector = vector.permute(0,2,1)  # (B,1,84)
    vector = self.conv1d_b(vector)  # (B, 168 , 1)
    vector = vector.squeeze(2)  # (B,168)
    channel_weights = self.sigmoid(vector)  # (B,168)    
    # Multiply the image and vector along the channel dimension.
    # vector: (C,)          x: (B, C, H, W)
    output = x * channel_weights.unsqueeze(2).unsqueeze(3)  # (B, C, H, W)

    return output
#___________________________________________________________________________________________________________________

class SeparableConvBlock(nn.Module):
  def __init__(self, in_channel, out_channels, kernel_size = (3,3)):
    super(SeparableConvBlock, self).__init__()
    self.in_channels , self.out_channels = in_channel , out_channels
    self.kernel_size = kernel_size
    self.seperable_conv = self.get_model()

  def get_model(self):
    return nn.Sequential(
        nn.Conv2d(in_channels = self.in_channels , out_channels = self.out_channels , kernel_size = (1,1)), 
        nn.Conv2d(in_channels = self.out_channels , out_channels = self.out_channels , kernel_size = self.kernel_size , 
                  padding = 'same' , groups = self.out_channels), 
        nn.PReLU(), 
        nn.BatchNorm2d(self.out_channels)
        
    )
  def forward(self, x):
    # print('inside forward of seperable conv block ')
    return self.seperable_conv(x)
#___________________________________________________________________________________________________________________

class SqueezeBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(SqueezeBlock, self).__init__()
    self.in_channels, self.out_channels =  in_channels, out_channels
    self.model = self.get_model()

  def get_model(self):
    return nn.Sequential(
        nn.BatchNorm2d(self.in_channels), 
        nn.Conv2d(in_channels = self.in_channels , out_channels = self.out_channels , kernel_size = (1,1))
    )
  def forward(self, x):
   return self.model(x)
#___________________________________________________________________________________________________________________

class XceptionBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(XceptionBlock, self).__init__()
    self.in_channels , self.out_channels = in_channels , out_channels
    self.xception_model = self.get_model()
    self.conv1_1 = nn.Conv2d(in_channels = self.in_channels , out_channels = self.out_channels , kernel_size = (1,1) , stride = (2,2))

  def get_model(self):
    return nn.Sequential(
        SeparableConvBlock(in_channel = self.in_channels , out_channels = self.out_channels) ,
        nn.PReLU() ,
        SeparableConvBlock(in_channel = self.out_channels , out_channels = self.out_channels) ,
        nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2)) ,
    )
  def forward(self, x):
    a = self.conv1_1(x)
    b = self.xception_model(x)
    return a+b   # side branch + main branch
  
#___________________________________________________________________________________________________________________

class ResidualBlock(nn.Module):
  def __init__(self, in_channels , n = 3):
    super(ResidualBlock, self).__init__()
    # n : no. of seperable conv blocks in a residual block
    
    self.n = n 
    self.in_channels  = in_channels
    self.model = self.get_model()

  def get_model(self):
    self.sep_conv_blocks = [SeparableConvBlock(in_channel = self.in_channels , out_channels = self.in_channels) for i in range(self.n)]
    return  nn.Sequential(*self.sep_conv_blocks)

  def forward(self, x):
    return x + self.model(x)    # side branch + main branch
#___________________________________________________________________________________________________________________
# model = ResidualBlock(256)
# import torch
# print(model.model)
# (model.forward(torch.randn(1,256,124,140)))

