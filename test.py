from dataset import WeedCocoDetection
from torchvision import transforms
import torchvision
import torchvision.models.detection
from torchvision.models.detection import *
from model import fasterrcnn_mobilenet_v3_large_320_fpn
import torch
from torch.utils.data import DataLoader
from torch import utils

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
dataset = WeedCocoDetection(root='C:/Users/gator/Hackathon/UFAIDaysHackathon/weedimages/data',
    annFile='C:/Users/gator/Hackathon/UFAIDaysHackathon/weedimages/labels.json',
    transform=transforms.Compose([]),
    target_transform=transforms.Compose([]),
    transforms=None
)
data_loader = DataLoader(
 dataset, batch_size=2, shuffle=True, num_workers=4,
 collate_fn=utils.collate_fn)

# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions

# print(torchvision.__version__)
dataset = WeedCocoDetection(root='C:/Users/gator/Hackathon/UFAIDaysHackathon/weedimages/data',
annFile='C:/Users/gator/Hackathon/UFAIDaysHackathon/weedimages/labels.json',
transform=transforms.Compose([]),
target_transform=transforms.Compose([]),
transforms=None
)

for image, label in dataset:
    print(image, label)

model = fasterrcnn_mobilenet_v3_large_320_fpn()

