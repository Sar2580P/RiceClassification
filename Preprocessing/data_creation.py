import os, sys
import pandas as pd
import json
sys.path.append(os.getcwd())
from utils import *
from sklearn.model_selection import train_test_split
import random
random.seed(42)
from sklearn.model_selection import StratifiedKFold

def segment_images():
  class_to_id = {}
  id_to_class = {}
  ct = 0
  rootdir = "dataset_v1"
  final_dir_hsi = "Data/hsi"
  final_dir_rgb = "Data/rgb"
  df_rgb = pd.DataFrame(columns=['path', 'class_id']) 
  df_hsi = pd.DataFrame(columns=['path', 'class_id'])

  skip_ct = 0
  for dirpath, dirnames, _ in os.walk(rootdir):
    
      for dirname in dirnames:
          print(ct , dirname)
          path = os.path.join(dirpath, dirname)
          class_ = dirname
          if class_ not in class_to_id:
            class_to_id[class_] = ct
            id_to_class[ct] = class_
            ct += 1

          for dirpath_, _, filenames in os.walk(path):
            for file in filenames:
              img_path = os.path.join(dirpath_, file)

              if  file.lower().endswith(".jpg"):
                start=0
                if file[-5]=='2':
                  start = 72
                contour_images, count = create_cropping_jpg(img_path)
                
                for i, img in enumerate(contour_images):
                  cv2.imwrite(os.path.join(final_dir_rgb, '{ct}_{i}.png'.format(ct = ct-1 ,i=i+start)), img)
                  df_rgb.loc[len(df_rgb)] = ['{ct}-{i}.png'.format(ct = ct-1 ,i=i+start), ct-1]

              elif  file.lower().endswith(".bil"):
                start=0
                if file[-5]=='2':
                  start = 72
                img = read_hdr(img_path)
                img = np.array(img.load())
                images = split_image(img, 25, 75, 700, 280, 12, 6)
                
                  
                for i, seed_image in enumerate(images):
                  name = '{}_{}.npy'.format(ct-1, start+i)
                  try:
                    np.save(os.path.join(final_dir_hsi, name), seed_image)
                    df_hsi.loc[len(df_hsi)] = [name, ct-1]
                  except:
                    print("error in writing hsi images", file)
                    skip_ct += 1
             
                                            
  df_rgb.to_csv('Data/rgb.csv', index = False)
  df_hsi.to_csv('Data/hsi.csv', index = False)
  print("\n\nskipped ", skip_ct, " images") 
  mappings = {'class_to_id': class_to_id, 'id_to_class': id_to_class}

  j = json.dumps(mappings, indent=4)
  with open('Data/mappings.json', 'w') as f:
      print(j, file=f)  

segment_images()
#__________________________________________________________________________________________________________________

class Preprocess():
  def __init__(self ,base_path_rgb , base_path_hsi , df_rgb , df_hsi, class_ct = 107):
    self.base_path_rgb = base_path_rgb
    self.base_path_hsi = base_path_hsi
    self.df_rgb = df_rgb
    self.df_hsi = df_hsi
    self.class_ct = class_ct
    self.dir = 'Data/{ct}'.format(ct = self.class_ct)
    if not os.path.exists(self.dir):
      os.mkdir(self.dir)

  def concat_df(self):
    self.df_final = pd.DataFrame(columns=['rgb_path', 'hsi_path' , 'class_id'])
    for i in range(len(self.df_rgb)):
      if self.df_rgb.iloc[i,1]>= self.class_ct:
         print('taken only {} classes'.format(self.class_ct))
         break
      if self.df_rgb.iloc[i,0][:-4] != self.df_hsi.iloc[i,0][:-4]:
        print('error in concat_df')
        break

      rgb_path = self.base_path_rgb +'/'+ self.df_rgb.iloc[i,0]
      hsi_path = self.base_path_hsi +'/'+ self.df_hsi.iloc[i,0]
      class_id = self.df_rgb.iloc[i,1]
      self.df_final.loc[len(self.df_final)] = [rgb_path , hsi_path , class_id]

    self.df_final.to_csv(os.path.join(self.dir ,'df_final.csv') , index = False)

  def split_df(self, df,  test_size = 0.2):
    df_train , df_test = train_test_split(df, test_size=test_size, stratify=df.iloc[:,2])
    df_test.to_csv(os.path.join(self.dir, 'df_tst.csv') , index = False)

    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    y = df_train.iloc[:,2].to_numpy()
    for fold, (train_index, val_index) in enumerate(skf.split(df_train, y)):
      if not os.path.exists(os.path.join(self.dir,'fold_{x}'.format(x = fold))):
        os.mkdir(os.path.join(self.dir,'fold_{x}'.format(x = fold)))

      df_train_fold, df_val_fold = df_train.iloc[train_index, :], df_train.iloc[val_index, :]
      
      df_train_fold.to_csv(os.path.join(self.dir,'fold_{x}'.format(x = fold) , 'df_tr.csv') , index = False)
      df_val_fold.to_csv(os.path.join(self.dir,'fold_{x}'.format(x = fold),  'df_val.csv') , index = False)
#__________________________________________________________________________________________________________________
# class_ct = 107
# p = Preprocess('Data/rgb', 'Data/hsi', pd.read_csv('Data/rgb.csv'), pd.read_csv('Data/hsi.csv'), class_ct)
# p.concat_df() 
# p.split_df(pd.read_csv('Data/{x}/df_final.csv'.format(x = class_ct)))
