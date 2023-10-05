import os, sys
import pandas as pd
import json
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from utils import *
from sklearn.model_selection import train_test_split
import random
random.seed(42)

def segment_images():
  class_to_id = {}
  id_to_class = {}
  ct = 0
  rootdir = "dataset_v1"
  final_dir = "Data/hsi"

  df = pd.DataFrame(columns=['image_name', 'class_id'])
  skip_ct = 0
  for dirpath, dirnames, _ in os.walk(rootdir):
    
      for dirname in dirnames:
          path = os.path.join(dirpath, dirname)
          class_ = dirname
          if class_ not in class_to_id:
            class_to_id[class_] = ct
            id_to_class[ct] = class_
            ct += 1

          for dirpath_, _, filenames in os.walk(path):
            for file in filenames:
                if not file.lower().endswith(".bil"):
                    continue
                img_path = os.path.join(dirpath_, file)
                start=0
                if file[-5]=='2':
                  start = 72
                img = read_hdr(img_path)
                img = np.array(img.load())
                images = split_image(img, 25, 75, 700, 280, 12, 6)
                
                  
                for i, seed_image in enumerate(images):
                  name = '{}_{}.npy'.format(ct-1, start+i)
                  try:
                    # cv2.imwrite(os.path.join(final_dir, name), seed_image)
                    np.save(os.path.join(final_dir, name), seed_image)
                  except:
                    print("error in writing ", file)
                    
                  df.loc[len(df)] = [name, ct-1]
              
                  
          

  print("\n\nskipped ", skip_ct, " images")      
  df.to_csv("Data/hsi.csv", index=False)
  mappings = {'class_to_id': class_to_id, 'id_to_class': id_to_class}

  j = json.dumps(mappings, indent=4)
  with open('Data/mappings.json', 'w') as f:
      print(j, file=f)  

#__________________________________________________________________________________________________________________

class Preprocess():
  def __init__(self ,base_path_rgb , base_path_hsi , df_rgb , df_hsi):
    self.base_path_rgb = base_path_rgb
    self.base_path_hsi = base_path_hsi
    self.df_rgb = df_rgb
    self.df_hsi = df_hsi

  def concat_df(self, class_ct = 107):
    self.df_final = pd.DataFrame(columns=['rgb_path', 'hsi_path' , 'class_id'])
    for i in range(len(self.df_rgb)):
      if self.df_rgb.iloc[i,1]>= class_ct:
         print('taken only {} classes'.format(class_ct))
         break
      if self.df_rgb.iloc[i,0][:-4] != self.df_hsi.iloc[i,0][:-4]:
        print('error in concat_df')
        break

      rgb_path = self.base_path_rgb +'/'+ self.df_rgb.iloc[i,0]
      hsi_path = self.base_path_hsi +'/'+ self.df_hsi.iloc[i,0]
      class_id = self.df_rgb.iloc[i,1]
      self.df_final.loc[len(self.df_final)] = [rgb_path , hsi_path , class_id]

    self.df_final.to_csv('Data/df_final.csv' , index = False)

  def split_df(self, df,  test_size = 0.2, val_size = 0.1):
    df_train , df_test = train_test_split(df, test_size=test_size, stratify=df.iloc[:,2])
    df_train ,df_val = train_test_split(df_train, test_size=val_size, stratify=df_train.iloc[:,2])

    df_train.to_csv('Data/df_tr.csv' , index = False)
    df_val.to_csv('Data/df_val.csv' , index = False)
    df_test.to_csv('Data/df_tst.csv' , index = False)

# p = Preprocess('Data/rgb', 'Data/hsi', pd.read_csv('Data/rgb.csv'), pd.read_csv('Data/hsi.csv'))
# p.concat_df() 
# p.split_df(pd.read_csv('Data/df_final.csv'))
