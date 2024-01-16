
import json
from tqdm import tqdm
import sys
sys.path.append('Preprocessing')
sys.path.append('models')
from train_eval import *
from model import *
from modules import *
from hsi.data_loading import tst_dataset as hsi_tst_dataset
from rgb.data_loading import tst_dataset as rgb_tst_dataset
from pytorch_lightning import Trainer
from dynamic_weighting import *
from utils import *

torch.set_float32_matmul_precision('high')

num_workers = 8
BATCH_SIZE = 64
hsi_tst_loader = DataLoader(hsi_tst_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
rgb_tst_loader = DataLoader(rgb_tst_dataset, batch_size=BATCH_SIZE,  shuffle=False, num_workers=num_workers)

#_______________________________________________________________________________________________________________________
hsi_densenet_config = 'models/hsi/dense_net/config.yaml'
hsi_densenet_model_obj = DenseNet(densenet_variant = [12, 18, 24, 6] , in_channels=168, num_classes=98 , 
                                  compression_factor=0.3 , k = 32 , config= load_config(hsi_densenet_config))
hsi_dense_net_ckpt ='weights/168_dense_net--epoch=102-val_loss=0.63-val_accuracy=0.79.ckpt'
hsi_densenet_model = Classifier.load_from_checkpoint(hsi_dense_net_ckpt, model_obj=hsi_densenet_model_obj)
hsi_densenet_trainer = Trainer(accelerator='gpu') 

hsi_densenet_trainer.test(hsi_densenet_model, hsi_tst_loader)

#_______________________________________________________________________________________________________________________
# rgb_gnet_config = 'models/rgb/google_net/config.yaml'
# rgb_gnet_model_obj = GoogleNet(load_config(rgb_gnet_config))
# rgb_gnet_ckpt = 'models/rgb/google_net/ckpts/gnet--epoch=44-val_loss=0.88-val_accuracy=0.77.ckpt'
# rgb_gnet_model = Classifier.load_from_checkpoint(rgb_gnet_ckpt, model_obj=rgb_gnet_model_obj)
# rgb_gnet_trainer = Trainer(accelerator='gpu')

# rgb_gnet_trainer.test(rgb_gnet_model, rgb_tst_loader)

#_______________________________________________________________________________________________________________________
# rgb_dnet_config = 'models/rgb/densenet/config.yaml'
# rgb_dnet_model_obj = DenseNetRGB(load_config(rgb_dnet_config))
# rgb_dnet_ckpt = 'models/rgb/densenet/ckpts/dnet--epoch=52-val_loss=0.76-val_accuracy=0.77.ckpt'
# rgb_dnet_model = Classifier.load_from_checkpoint(rgb_dnet_ckpt, model_obj=rgb_dnet_model_obj)
# rgb_dnet_trainer = Trainer(accelerator='gpu')

# rgb_dnet_trainer.test(rgb_dnet_model, rgb_tst_loader)
#______________________________________________

# rgb_mnet_config = 'models/rgb/densenet/config.yaml'
# rgb_mnet_model_obj = MobileNet(load_config(rgb_mnet_config))
# rgb_mnet_ckpt = 'models/rgb/densenet/ckpts/dnet--epoch=52-val_loss=0.76-val_accuracy=0.77.ckpt'
# rgb_mnet_model = Classifier.load_from_checkpoint(rgb_mnet_ckpt, model_obj=rgb_mnet_model_obj)
# rgb_mnet_trainer = Trainer(accelerator='gpu')

# rgb_mnet_trainer.test(rgb_mnet_model, rgb_tst_loader) 
#_________________________________________________________________________

def get_logit_df(model, model_name):
  result = pd.DataFrame(columns = [str(i) for i in range(98)])
  labels = []
  for batch in model.y_true:
    labels.extend(list(batch.detach().cpu().numpy()))
    
  for batch in model.y_hat:
    batch = batch.detach().cpu().numpy()
    for i in range(batch.shape[0]):
      result.loc[len(result)] = batch[i]
  
  result['labels'] = labels
  result.to_csv('models/ensemble/Classifier_Prediction_Data/98/{model_name}.csv'.format(model_name = model_name) , index = False)
  return result


 
 
#_______________________________________________________________________________________________________________________

# x = get_logit_df(hsi_densenet_model, 'denseNet_hsi_classifier')
# p1 , labels = x.iloc[: , :-1] , x.iloc[: , -1]


# x = get_logit_df(rgb_dnet_model, 'denseNet_rgb_classifier')
# p2 , labels = x.iloc[: , :-1] , x.iloc[: , -1]

n = 98
df_hsi = pd.read_csv('models/ensemble/Classifier_Prediction_Data/98/168_denseNet_hsi_classifier.csv')
# df_hsi = pd.read_csv('models/ensemble/Classifier_Prediction_Data/{n}/denseNet_hsi_classifier.csv'.format(n=n))
df_rgb = pd.read_csv('models/ensemble/Classifier_Prediction_Data/{n}/denseNet_rgb_classifier.csv'.format(n=n))

p1 , p2,  labels = df_hsi.iloc[: , :n] , df_rgb.iloc[:, :n] , df_hsi.iloc[: , n]

#_______________________________________________________________________________________________________________________

# top = 30    #top 'k' classes
# predictions = Gompertz(top=top, argv = (p1, p2))

# correct = np.where(predictions == labels)[0].shape[0]
# total = labels.shape[0]

# print("Accuracy = ",correct/total)
# classes = []
# for i in range(p1.shape[1]):
#     classes.append(str(i+1))
    
# metrics(labels,predictions,classes)

# plot_roc(labels,predictions)
#_______________________________________________________________________________________________________________________


arr = np.arange(0, 1.005, 0.005)
y_true = labels
y_predicted = None
accuracies = []
# # w1 = 0.68
# # 
max = 0
max_lambda = 0
for w1 in tqdm(arr):
  preds = []
  for i in range(len(labels)) :
    p = (w1 * p1.iloc[i , :]) + ((1-w1) * p2.iloc[i , :])
    output_class = np.argmax(p, axis =0)
    preds.append(output_class)

  accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
  if accuracy > max:
    max = accuracy
    max_lambda = w1
    y_predicted = preds
  accuracies.append(round(accuracy , 4))
  

plt.plot(arr , accuracies)
plt.xlim(0,1)
plt.ylim(0,1)
plt.vlines(max_lambda ,0 , max , colors='r' , linestyles='dashed')
plt.hlines(max , 0 , max_lambda , colors='r' , linestyles='dashed')
plt.xlabel('lambda')
plt.ylabel('ensemble_accuracy')
plt.savefig('models/ensemble/Classifier_Prediction_Data/ensemble_accuracy.pdf')

print("Max accuracy = ", max)
print("Max lambda = ", max_lambda)

# get confusion matrix
json_file = json.load(open('Data/mappings.json'))
classes = list(json_file['class_to_id'].keys())

from sklearn.metrics import confusion_matrix
for i in range(len(y_predicted)):
  y_predicted[i] = json_file['id_to_class'][str(int(y_predicted[i]))]
  y_true[i] = json_file['id_to_class'][str(int(y_true[i]))]  
  
cm = confusion_matrix(y_true, y_predicted, labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.to_csv('models/ensemble/Classifier_Prediction_Data/confusion_matrix.csv')