from model import WeedDetection
import pytorch_lightning as pl

net = WeedDetection()
trainer = pl.Trainer(max_epochs=5, enable_progress_bar=True)
trainer.fit(net)