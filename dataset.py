import typing
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

class WeedCocoDetection(torchvision.datasets.CocoDetection):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

Args:
    root (string): Root directory where images are downloaded to.
    annFile (string): Path to json annotation file.
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    transforms (callable, optional): A function/transform that takes input sample and its target as entry
        and returns a transformed version."""
    def __init__(self, root: str = "C:/Users/gator/Hackathon/UFAIDaysHackathon/weedimages/data", 
        annFile: str = "C:/Users/gator/Hackathon/UFAIDaysHackathon/weedimages/labels.json", 
        transform: typing.Optional[typing.Callable] = None, 
        target_transform: typing.Optional[typing.Callable] = None, 
        transforms: typing.Optional[typing.Callable] = None):
        super().__init__(root, annFile, transform, target_transform, transforms)

    def __getitem__(self, idx):
        image, annotations = super().__getitem__(idx)
        boxes = []
        labels = []
        areas = []
        iscrowds = []
        image_ids = []
        for annotation in annotations:
            box = [annotation["bbox"][0], annotation["bbox"][1], annotation["bbox"][0]+annotation["bbox"][2], annotation["bbox"][1]+annotation["bbox"][3]]
            label = annotation["category_id"]
            area = annotation["area"]
            iscrowd = annotation["iscrowd"]
            image_id = annotation["image_id"]
            boxes.append(box)
            labels.append(label)
            areas.append(area)
            iscrowds.append(iscrowd)
            image_ids.append(image_id)
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        if(len(boxes)<=0):
             target["boxes"] = torch.zeros((0,4),dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.FloatTensor(image_ids)
        target["area"] = torch.FloatTensor(areas)
        target["iscrowd"] = torch.FloatTensor(iscrowds)
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)
        return image, target
