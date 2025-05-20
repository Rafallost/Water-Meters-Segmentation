import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.model import WaterMetersUNet
from dataset import WMSDataset
from transforms import imageTransforms
import os

# Preparing dataloader
baseDataDir = os.path.join(os.path.dirname(__file__), '..', 'data')
def gather_paths(split):
    img_dir = os.path.join(baseDataDir, split, 'images')
    mask_dir = os.path.join(baseDataDir, split, 'masks')
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
    masks  = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.jpg')])
    return images, masks

_, _      = gather_paths('train')  # Only init
testImgs, testMasks = gather_paths('test')
testDataset = WMSDataset(testImgs, testMasks, imageTransforms)
testLoader  = DataLoader(testDataset, batch_size=4, shuffle=False)

# Loading model and weight
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WaterMetersUNet(inChannels=3, outChannels=1).to(device)
checkpoint = torch.load('../models/best.pth', map_location=device) # CHOOSE CHECKPOINT
model.load_state_dict(checkpoint)
model.eval()

# N = number of samples
N = 12
collected = 0
with torch.no_grad():
    for images, masks in testLoader:
        images = images.to(device)
        probs  = torch.sigmoid(model(images))
        preds  = (probs > 0.5).float().cpu().numpy()
        imgs_np = images.cpu().permute(0,2,3,1).numpy()
        masks_np = masks.cpu().squeeze(1).numpy()

        B = images.shape[0]
        for i in range(B):
            if collected >= N:
                break
            fig, axes = plt.subplots(1,3, figsize=(12,4))
            axes[0].imshow(imgs_np[i])
            axes[0].set_title('Image')
            axes[0].axis('off')
            axes[1].imshow(masks_np[i], cmap='gray')
            axes[1].set_title('GT Mask')
            axes[1].axis('off')
            axes[2].imshow(preds[i].squeeze(), cmap='gray')
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            plt.show()
            collected += 1
        if collected >= N:
            break
