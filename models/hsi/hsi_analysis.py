import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from sympy import plot
import seaborn as sns

BASE_PATH = 'Data/hsi'
df = pd.read_csv('Data/hsi.csv')
data = df.sample(frac = 0.3 , random_state=42 , replace = False) 

def get_pixel_distribution(img_channel):
    pixel_values = img_channel.reshape(-1)
    return np.mean(pixel_values),  np.std(pixel_values)


def full_distribution(data):
  distribution_df = pd.DataFrame(columns=['channel' ,'mean', 'std']) 
  j = 0   
  for i in range(len(data)):
      img = np.load(os.path.join(BASE_PATH , data.iloc[i ,0]))
      for channel in range(img.shape[2]):
        mean, std = get_pixel_distribution(img[:,:,channel])
        distribution_df.loc[j] = [channel ,mean, std]
        j+=1
  distribution_df.to_csv('models/hsi/hsi_distribution.csv', index=False)

# full_distribution(data)
df = pd.read_csv('models/hsi/hsi_distribution.csv')


def plot_distribution(df):
  print(123)
  df['channel'] = df['channel'].astype(int)
  channels = df['channel'].unique()
  print(channels)

  fig, axs = plt.subplots(42, 4, figsize=(30, 200))
  fig.tight_layout(pad=7.0)
  for i , channel in enumerate(channels):
    print(i)
    r,c = i//4 , i%4
    channel_df = df[df['channel'] == channel]
    axs[r,c].scatter(channel_df['mean'], channel_df['std'], color = 'red', marker = 'o')
    axs[r,c].set_title(f'Channel-{channel}')
    axs[r,c].set_xlabel('mean')
    # axs[r,c].set_xticks(rotate = 90)
    axs[r,c].set_ylabel('std')
  plt.savefig('models/hsi/mean_distribution.png')

plot_distribution(df)
# print(df.shape)