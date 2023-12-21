import sys, argparse
from tqdm import tqdm
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

hsi_pretrained = nn.Sequential(*list(hsi_classifier.model.children()))
rgb_pretrained = nn.Sequential(*list(rgb_classifier.model.children()))
# df_tst = pd.read_csv('Data/96/df_tst.csv')
df_tr = pd.read_csv('Data/96/fold_0/df_tr.csv')

for param in hsi_pretrained.parameters():
  param.requires_grad = False

for param in rgb_pretrained.parameters():
  param.requires_grad = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#_______________________________________________________________________________________________________________________
def get_prediction(data_df ,classifier, classifier_name, input_type,  transforms_):
  labels = data_df.iloc[: ,2]
  df = pd.DataFrame(columns = [str(i) for i in range(96)])
  
  for idx in tqdm(range(len(data_df))):
    img = None
    if input_type == 'hsi':
      img = np.load(data_df.iloc[idx , 1])
    else :
      img = cv2.imread(data_df.iloc[idx , 0])
    img = torch.unsqueeze(transforms_(img), 0)
    output = classifier(img.to(device))
    output = output.detach().cpu().numpy()
    df.loc[len(df)] = output[0]

  df['labels'] = labels
   
  df.to_csv('models/ensemble/Classifier_Prediction_Data/'+ classifier_name + '.csv' , index = False)
  return df, labels

p1, labels = get_prediction(df_tr ,hsi_pretrained, 'denseNet_hsi_classifier', 'hsi',transforms.ToTensor())
p2, _ = get_prediction(df_tr ,rgb_pretrained, 'googleNet_rgb_classifier', 'rgb',transforms.ToTensor())

df_hsi = pd.read_csv('models/ensemble/Classifier_Prediction_Data/denseNet_hsi_classifier.csv')
df_rgb = pd.read_csv('models/ensemble/Classifier_Prediction_Data/googleNet_rgb_classifier.csv')

p1 , p2 , labels = df_hsi.iloc[: , :96] , df_rgb.iloc[: , :96] , df_hsi.iloc[: , 96]
# parser = argparse.ArgumentParser()
# args = parser.parse_args()

# parser.add_argument('--topk', type=int, default = 2, help='Top-k number of classes')

# top = 3#top 'k' classes
# predictions = Gompertz(top, p1, p2)

# correct = np.where(predictions == labels)[0].shape[0]
# total = labels.shape[0]

# print("Accuracy = ",correct/total)
# classes = []
# for i in range(p1.shape[1]):
#     classes.append(str(i+1))
    
# metrics(labels,predictions,classes)

# plot_roc(labels,predictions)
#_______________________________________________________________________________________________________________________


w1 = 1.0
preds = []

for i in range(len(labels)) :
  p = (w1 * p1.iloc[i , :]) + ((1-w1) * p2.iloc[i , :])
  output_class = np.argmax(p)
  preds.append(output_class)
  
print(len(preds))
accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
print(accuracy)