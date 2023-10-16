from modules import *
import torch 

class HSIModel(nn.Module):
  def __init__(self , in_channels, num_classes):
    super(HSIModel, self).__init__()
    self.bam = BandAttentionBlock(in_channels)
    self.squeeze_channels = 100
    self.squeeze = SqueezeBlock(in_channels, self.squeeze_channels)    # in_channels = 168
    self.xception1 = XceptionBlock(self.squeeze_channels, 128)
    self.xception2 = XceptionBlock(128, 256)
    self.residual = ResidualBlock(256, 256)
    self.xception3 = XceptionBlock(256, 512)
    self.sep_conv1 = SeparableConvBlock(512, 1024)
    self.max_pool = nn.MaxPool2d(kernel_size = (3,3) , stride = (2,2))
    self.gap = nn.AdaptiveAvgPool2d((1,1))

    self.fc1 = Dense(0.2 , 1024, 256)
    self.fc2 = Dense(0, 256, num_classes)

  def forward(self, x):
    x = self.bam(x)
    print('1 : ' , x.shape)
    x = self.squeeze(x)
    print('2 : ' , x.shape)
    x = self.xception1(x)
    print('3 : ' , x.shape)
    x = self.xception2(x)
    print('4 : ' , x.shape)
    x = self.residual(x)
    print('5 : ' , x.shape)
    x = self.xception3(x)
    print('6 : ' , x.shape)
    x = self.sep_conv1(x)
    print('7 : ' , x.shape)
    x = self.max_pool(x)
    print('8 : ' , x.shape)
    x = self.gap(x)
    print('9 : ' , x.shape)
    x = torch.flatten(x, 1)
    print('10 : ' , x.shape)
    x = self.fc1(x)
    print('11 : ' , x.shape)
    x = self.fc2(x)
    print('12 : ' , x.shape)
    return x
  

model = HSIModel(168, 107)

x = torch.randn(32,168, 256, 256)
y = model(x)
    