import typing
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import *
from dataset import WeedCocoDetection
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

def collate_fn(batch):
    return tuple(zip(*batch))

class WeedDetection(pl.LightningModule):

    def __init__(self, weights: typing.Optional[torchvision.models.detection.ssdlite.SSDLite320_MobileNet_V3_Large_Weights] = None, progress: bool = True, num_classes: typing.Optional[int] = None, weights_backbone: typing.Optional[torchvision.models.MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1, trainable_backbone_layers: typing.Optional[int] = None, norm_layer: typing.Optional[typing.Callable[..., torch.nn.modules.module.Module]] = None, **kwargs: typing.Any):
        super(WeedDetection, self).__init__()
        self.learning_rate = 1e-3
        self.batch_size = 4
        self.model=torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights, progress, num_classes, weights_backbone, trainable_backbone_layers, norm_layer, **kwargs)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def prepare_data(self):
        self.train_dataset = WeedCocoDetection(root='C:/Users/gator/Hackathon/UFAIDaysHackathon/weedimages/data',
            annFile='C:/Users/gator/Hackathon/UFAIDaysHackathon/weedimages/labels.json',
            transform=transforms.Compose([]),
            target_transform=transforms.Compose([]),
            transforms=None
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('Loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.95, weight_decay=1e-5, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0, verbose=True)
        return [optimizer], [scheduler]