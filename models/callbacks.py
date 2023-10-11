from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from torchvision import transforms

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=5,
   verbose=True,
   mode='min'
)
rich_progress_bar = RichProgressBar(metrics='green', time='yellow', progress_bar_finished='#8c53e0' ,progress_bar='#c99e38')

rich_model_summary = RichModelSummary(max_depth=2)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=2,
    verbose=True,
 )

