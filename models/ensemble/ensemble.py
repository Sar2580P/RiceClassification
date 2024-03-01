
import enum
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
import pickle
torch.set_float32_matmul_precision('high')

fold = 4

#_________________________________________________________________________

folds = {
  0 : [
    'models/rgb/densenet/evaluations/rgb_denseNet_net__var-98__fold-0__predictions.pkl', 
    'models/hsi/dense_net/evaluations/hsi_densenet__var-98__fold-0__predictions.pkl' , 

  ], 
  1 : [
    'models/rgb/densenet/evaluations/rgb_denseNet_net__var-98__fold-1__predictions.pkl',
    'models/hsi/dense_net/evaluations/hsi_densenet__var-98__fold-1__predictions.pkl' , 

  ], 
  2: [
    'models/rgb/densenet/evaluations/rgb_denseNet_net__var-98__fold-2__predictions.pkl' ,
    'models/hsi/dense_net/evaluations/hsi_densenet__var-98__fold-2__predictions.pkl' , 
  ], 
  3 : [
    'models/rgb/densenet/evaluations/rgb_denseNet_net__var-98__fold-3__predictions.pkl',
    'models/hsi/dense_net/evaluations/hsi_densenet__var-98__fold-3__predictions.pkl' , 

  ], 
  4: [
    'models/rgb/densenet/evaluations/rgb_denseNet_net__var-98__fold-4__predictions.pkl' ,
    'models/hsi/dense_net/evaluations/hsi_densenet__var-98__fold-4__predictions.pkl' , 
  ]

}

def get_logit_df(fold):
  rgb_path , hsi_path = folds[fold][0], folds[fold][1]
  model_names = ['rgb_denseNet_net', 'hsi_densenet']

  for _, (model_name, path) in enumerate(zip(model_names, [rgb_path, hsi_path])):

    file = pickle.load(open(path, 'rb'))
    y_hat, y_true = file['y_hat'] , file['y_true']

    result = pd.DataFrame(columns = [str(i) for i in range(98)])
    labels = []
    for batch in y_true:
      labels.extend(list(batch.detach().cpu().numpy()))
      
    for batch in y_hat:
      batch = batch.detach().cpu().numpy()
      for i in range(batch.shape[0]):
        result.loc[len(result)] = batch[i]
    
    result['labels'] = labels
    result.to_csv('models/ensemble/Classifier_Prediction_Data/98/{model_name}--{fold}.csv'.format(fold = fold, model_name = model_name) , index = False)
  return

# for fold in range(5):
#   get_logit_df(fold)
 

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

def get_ensemble_performance():
  arr = np.arange(0, 1.005, 0.005)
  ensemble_fold_stats = {}

  for fold in range(5):
    p1 = pd.read_csv('models/ensemble/Classifier_Prediction_Data/98/hsi_densenet--{fold}.csv'.format(fold = fold))
    p2 = pd.read_csv('models/ensemble/Classifier_Prediction_Data/98/rgb_denseNet_net--{fold}.csv'.format(fold = fold))
    labels = p1['labels']
    p1 = p1.iloc[: , :98]
    p2 = p2.iloc[: , :98]

    y_true, y_predicted = labels, None
    accuracies = []
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

    ensemble_fold_stats[fold] = {
      'max_accuracy' : max,
      'max_lambda' : max_lambda,
      'y_true' : y_true,
      'y_predicted' : y_predicted, 
      'accuracies' : accuracies
    }
  pickle.dump(ensemble_fold_stats, open('models/ensemble/Classifier_Prediction_Data/ensemble_fold_stats.pkl', 'wb'))

    
# get_ensemble_performance()






# # plt.plot(arr , accuracies)
# # plt.xlim(0,1)
# # plt.ylim(0,1)
# # plt.vlines(max_lambda ,0 , max , colors='r' , linestyles='dashed')
# # plt.hlines(max , 0 , max_lambda , colors='r' , linestyles='dashed')
# # plt.xlabel('lambda')
# # plt.ylabel('ensemble_accuracy')
# # plt.savefig('models/ensemble/Classifier_Prediction_Data/ensemble_accuracy.pdf')


# # get confusion matrix
# json_file = json.load(open('Data/mappings.json'))
# classes = list(json_file['class_to_id'].keys())

# # from sklearn.metrics import confusion_matrix
# # for i in range(len(y_predicted)):
# #   y_predicted[i] = json_file['id_to_class'][str(int(y_predicted[i]))]
# #   y_true[i] = json_file['id_to_class'][str(int(y_true[i]))]  
  
# # cm = confusion_matrix(y_true, y_predicted, labels=classes)
# # cm_df = pd.DataFrame(cm, index=classes, columns=classes)
# # cm_df.to_csv('models/ensemble/Classifier_Prediction_Data/confusion_matrix.csv')