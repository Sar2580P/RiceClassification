from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from torchvision import transforms

from  pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.0001,
   patience=20,
   verbose=True,
   mode='min'
)

theme = RichProgressBarTheme(metrics='green', time='yellow', progress_bar_finished='#8c53e0' ,progress_bar='#c99e38')
rich_progress_bar = RichProgressBar(theme=theme)

rich_model_summary = RichModelSummary(max_depth=5)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=3,
    verbose=True,
 )
rgb_transforms = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.32),
      transforms.RandomVerticalFlip(p=0.32),
      transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.2)),
      transforms.ToTensor(),
   ])

hsi_img_transforms = transforms.Compose([
   transforms.ToTensor(), 
   transforms.RandomHorizontalFlip(p=0.32),
   transforms.RandomVerticalFlip(p=0.32),
   transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.2)),
   
])
