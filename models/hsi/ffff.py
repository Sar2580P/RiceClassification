hsi_ckpt = 'models/hsi/xception/ckpts/xception--epoch=1-val_loss=3.27-val_accuracy=0.15.ckpt'

import sys

sys.path.append('Preprocessing')
sys.path.append('models')
from utils import *
from train_eval import *
from model import HSIModel

hsi_obj = HSIModel(load_config('models/hsi/xception/config.yaml'))

hsi_classifier = Classifier.load_from_checkpoint(hsi_ckpt, model_obj=hsi_obj)