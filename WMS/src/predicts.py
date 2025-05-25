import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.model import WaterMetersUNet
from dataset import WMSDataset
from transforms import imageTransforms
import cv2
import os

# Preparing dataloader
baseDataDir = os.path.join(os.path.dirname(__file__), '..', 'data')

img_dir = os.path.join(baseDataDir, 'test', 'images')
mask_dir = os.path.join(baseDataDir, 'test', 'masks')
testImgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
testMasks  = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.jpg')])

custom_dir = os.path.join(baseDataDir, 'custom')
save_dir   = os.path.join(baseDataDir, 'custom_predictions')
os.makedirs(save_dir, exist_ok=True)

testDataset = WMSDataset(testImgs, testMasks, imageTransforms)
testLoader  = DataLoader(testDataset, batch_size=16, shuffle=False)

# Loading model and weight
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WaterMetersUNet(inChannels=3, outChannels=1).to(device)
checkpoint = torch.load('../models/best.pth', map_location=device) # CHOOSE CHECKPOINT
model.load_state_dict(checkpoint)
model.eval()

for fname in os.listdir(custom_dir):
    if not fname.lower().endswith('.jpg'):
        continue
    img_path = os.path.join(custom_dir, fname)
    # 1) Wczytaj i skonwertuj na RGB
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    orig_resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    # 2) Transformacje takie jak w treningu
    tensor = imageTransforms(rgb).unsqueeze(0).to(device)  # [1,3,512,512]
    # 3) Predykcja
    with torch.no_grad():
        out   = model(tensor)
        prob  = torch.sigmoid(out).squeeze().cpu().numpy()   # [512,512]
        predM = (prob > 0.5).astype('uint8') * 255           # 0/255

    # 4) Wyświetl obie siatki
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(orig_resized)
    ax[0].set_title('Original')
    ax[0].axis('off')
    ax[1].imshow(predM, cmap='gray')
    ax[1].set_title('Predicted mask')
    ax[1].axis('off')
    plt.suptitle(fname)
    plt.show()

    # 5) (opcjonalnie) zapisz maskę
    cv2.imwrite(os.path.join(save_dir, f"mask_{fname}"), predM)

# N = number of samples
N = 0
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
