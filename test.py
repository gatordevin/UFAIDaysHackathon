from model import WeedDetection
import pytorch_lightning as pl
import torch
import numpy
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
import matplotlib.patches as patches

torch.device("cuda")

image_dir = "images/"

if __name__ == '__main__':
    net : WeedDetection = WeedDetection.load_from_checkpoint("version_1/checkpoints/epoch=99-step=8600.ckpt")
    net.eval()
    for path in os.listdir(image_dir):
        image_path = image_dir+path
        image = Image.open(image_path)
        # np_img = np.array(image)
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)
        images = [image]
        # image.permute(2,0,1)
        preds = net.forward(images)
        for image, pred in zip(images,preds):
            fig, ax = plt.subplots()
            box_tensor = list(pred["boxes"])
            scores_tensor = list(pred["scores"])
            labels_tensor = list(pred["labels"])
            ax.imshow(image.cpu().permute(1,2,0))
            for score in scores_tensor:
                if(score>0.8):
                    box_index = scores_tensor.index(score)
                    box = box_tensor[box_index].cpu()
                    rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    print(box)
                    print(labels_tensor[box_index])
                    print(box_index)
            plt.show()
    # trainer = pl.Trainer(max_epochs=100,accelerator='gpu', devices=1, enable_progress_bar=True)
    # net.
    # trainer.test(model=net,ckpt_path="lightning_logs/version_1/checkpoints/epoch=99-step=8600.ckpt")