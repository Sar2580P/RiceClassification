import sys, argparse

from pyparsing import col
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
from dynamic_weighting import *
#_______________________________________________________________________________________________________________________

denseNet_obj = DenseNet(densenet_variant = [12, 18 ,24 , 12] , in_channels=152, num_classes=96 , 
                        compression_factor=0.3 , k = 32 , config=load_config('models/hsi/dense_net/config.yaml'))
gnet_obj = GoogleNet(load_config('models/rgb/google_net/config.yaml'))


hsi_ckpt = os.path.join('models/hsi/dense_net/ckpts' , os.listdir('models/hsi/dense_net/ckpts')[-1])
rgb_ckpt = os.path.join('models/rgb/google_net/ckpts' , os.listdir('models/rgb/google_net/ckpts')[-1])

hsi_classifier = Classifier.load_from_checkpoint(hsi_ckpt, model_obj=denseNet_obj)
rgb_classifier = Classifier.load_from_checkpoint(rgb_ckpt, model_obj=gnet_obj)


# hsi_pretrained = nn.Sequential(*list(hsi_classifier.model.children())[0][:-1])
# rgb_pretrained = nn.Sequential(*list(rgb_classifier.model.children())[:-1])

#_______________________________________________________________________________________________________________________
for param in hsi_classifier.parameters():
  param.requires_grad = False

for param in rgb_classifier.parameters():
  param.requires_grad = False

#_______________________________________________________________________________________________________________________
def get_prediction(data_df ,classifier, classifier_name, input_type,  transforms_):
  labels = data_df.iloc[: ,1]
  df = pd.DataFrame(columns = [str(i) for i in range(96)])
  
  for img_path in data_df.iloc[:,0]:
      img = None
      if input_type == 'hsi':
        img = np.load(img_path)

      else :
         img = cv2.imread(img_path)

      img = transforms_(img)
      print(img.shape)
      output = torch.softmax(classifier(img))
      output = output.detach().cpu().numpy()
      df.loc[len(df)] = output

  # df['labels'] = labels
   
  # df.to_csv('models\ensemble\Classifier_Prediction_Data', classifier_name + '.csv' , index = False)
  return df, labels

p1, labels = get_prediction(df_tst ,hsi_classifier, 'denseNet_hsi_classifier', 'hsi',transforms.ToTensor())
p2, _ = get_prediction(df_tst ,rgb_classifier, 'googleNet_rgb_classifier', 'rgb',transforms.ToTensor())
parser = argparse.ArgumentParser()
args = parser.parse_args()

parser.add_argument('--topk', type=int, default = 2, help='Top-k number of classes')

top = args.topk #top 'k' classes
predictions = Gompertz(top, p1, p2)

correct = np.where(predictions == labels)[0].shape[0]
total = labels.shape[0]

print("Accuracy = ",correct/total)
classes = []
for i in range(p1.shape[1]):
    classes.append(str(i+1))
    
metrics(labels,predictions,classes)

plot_roc(labels,predictions)
#_______________________________________________________________________________________________________________________



