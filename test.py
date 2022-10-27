from model import WeedDetection
import pytorch_lightning as pl
import torch

torch.device("cuda")

if __name__ == '__main__':
    net = WeedDetection()
    trainer = pl.Trainer(max_epochs=100,accelerator='gpu', devices=1, enable_progress_bar=True)
    trainer.test(model=net,ckpt_path="lightning_logs/version_1/checkpoints/epoch=99-step=8600.ckpt")