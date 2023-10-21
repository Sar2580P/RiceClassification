import  sys
sys.path.append('models')
sys.path.append('Preprocessing')
sys.path.append('models/rgb')
from train_eval import *
from model import EffecientNet
from callbacks import *
from utils import *
from pytorch_lightning import Trainer
from data_loading import *
import pickle

torch.set_float32_matmul_precision('high')

config_path = 'models/rgb/enet/config.yaml'
config = load_config(config_path)

model_obj = EffecientNet(config)

num_workers = 8
model = Classifier(model_obj)

trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary], 
                  max_epochs=25)  
path = 'Data/107'
test_results = []

for i in range(5):
  tr_path = os.path.join(path, 'fold_{x}'.format(x=i) ,'df_tr.csv')
  val_path = os.path.join(path,'fold_{x}'.format(x=i) ,'df_val.csv')
  tst_path = os.path.join(path, 'df_tst.csv')
  df_tr = pd.read_csv(tr_path).iloc[:,[0,2]]
  tr_dataset = MyDataset(df_tr, img_transforms)

  df_val = pd.read_csv(val_path).iloc[:,[0,2]]
  val_dataset = MyDataset(df_val, img_transforms)

  df_tst = pd.read_csv(tst_path).iloc[:,[0,2]]
  tst_dataset = MyDataset(df_tst, img_transforms)

  tr_loader = DataLoader(tr_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=num_workers)
  val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)
  tst_loader = DataLoader(tst_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=num_workers)

  trainer.fit(model, tr_loader, val_loader)
  x = trainer.test(model, tst_loader)
  test_results.append(x)
  print('Completed {x}/5  folds'.format(x=(i+1)) , end='\n\n')
  
print(test_results)

pickle.dump(test_results, open(os.path.join('models/rgb/enet', 'sk_fold_test_results.pkl'), 'wb'))
