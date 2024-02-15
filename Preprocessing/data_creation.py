import os, sys
import pandas as pd
import json
sys.path.append(os.getcwd())
from utils import *
# from sklearn.model_selection import train_test_split
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

# segment_images()
#__________________________________________________________________________________________________________________


def create_folds(final_df, variety_count,  num_folds = 5):
  BASE_PATH = 'Data/'+str(variety_count)
  mapping_json = json.load(open(os.path.join(BASE_PATH, f'{variety_count}_var_mappings.json')))

  class_labels = [int(x) for x in mapping_json['id_to_class'].keys()]
  mask = final_df['class_id'].isin(class_labels)
  df = final_df[mask]
  df = df.reset_index(drop=True)

  skf = StratifiedKFold(n_splits=num_folds, random_state=42, shuffle=True)
  for fold, (train_index, val_index) in enumerate(skf.split(df, df.loc[:, 'class_id'])):

    if not os.path.exists(os.path.join(BASE_PATH, f'fold_{fold}')):
      os.mkdir(os.path.join(BASE_PATH, f'fold_{fold}'))

    df_train_fold, df_val_fold = df.iloc[train_index, :], df.iloc[val_index, :]
    
    df_train_fold.to_csv(os.path.join(BASE_PATH, f'fold_{fold}' , 'df_tr.csv') , index = False)
    df_val_fold.to_csv(os.path.join(BASE_PATH, f'fold_{fold}' ,  'df_val.csv') , index = False)
    print('fold_{x} created'.format(x = fold))


if not os.path.exists('Data/final_df.csv'):
  hsi_df = pd.read_csv('Data/hsi.csv')

  RGB_BASE_PATH, HSI_BASE_PATH = 'Data/rgb' , 'Data/hsi'
  final_df = pd.DataFrame(columns=['rgb_path', 'hsi_path' , 'class_id'])
  for i in range(len(hsi_df)):

    rgb_path = RGB_BASE_PATH +'/'+ hsi_df.iloc[i,0][:-4]+'.png' 
    hsi_path = HSI_BASE_PATH +'/'+ hsi_df.iloc[i,0]
    class_id = hsi_df.iloc[i,1]
    final_df.loc[len(final_df)] = [rgb_path , hsi_path , class_id]

  final_df.to_csv('Data/final_df.csv' , index = False)



# li = [12,24,37,55,75, 98]
# for variety_count in li:
#   create_folds(pd.read_csv('Data/final_df.csv'), variety_count)
#   print('done for ', variety_count)