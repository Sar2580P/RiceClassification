import os
import matplotlib.pyplot as plt 
import numpy as np

y = np.load('Data/hsi/0_62.npy')
img = y

fig, axs = plt.subplots(10, 20, figsize=(10, 5))

d = min(y.shape[2]-1 , 200)
for i in range(200):
    ax = axs[i//20, i%20]
    j = min(i, d)
    ax.imshow(y[:,:,j], cmap="gray")
    ax.axis('off')

# plt.imsave('wefcweferfreg.png', img, cmap="gray")
plt.show()