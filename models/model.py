from torchvision import transforms
import torch.nn as nn 
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Resnet():
  def __init__(self, config):
    self.config = config
    self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True).to(device)
    self.transforms_ = transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          ])
    self.model = self.__create_model__()

  def __create_model__(self):
    for param in self.resnet.parameters():
      param.requires_grad = False 

    self.resnet.fc = nn.Sequential(
                Dense(0.2 , 2048 ,1024), 
                Dense(0.15, 1024, 512) ,
                Dense(0, 512, self.config['num_classes']) 
                        ).to(device)
    return self.resnet
    
  def forward(self, x):
    return self.model(x) 


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