import subprocess
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary
from scipy.spatial.distance import directed_hausdorff
from src.model import WaterMetersUNet
from dataset import WMSDataset
from transforms import imageTransforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Dice coefficient
def dice_coeff(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Intersection over Union
def iou_coeff(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Pixel-wise accuracy
def pixel_accuracy(pred, target):
    return (pred == target).mean()

# Prepare data
prepare_script = os.path.join(os.path.dirname(__file__), 'prepareDataset.py')
subprocess.run([sys.executable, prepare_script], check=True)

# Load data
baseDataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
# Utility to gather paths
def gather_paths(split):
    img_dir = os.path.join(baseDataDir, split, 'images')
    mask_dir = os.path.join(baseDataDir, split, 'masks')
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
    masks  = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.jpg')])
    return images, masks

trainImagePaths, trainMaskPaths = gather_paths('train')
testImagePaths, testMaskPaths   = gather_paths('test')
valImagePaths,  valMaskPaths    = gather_paths('val')

trainDataset = WMSDataset(trainImagePaths, trainMaskPaths, imageTransforms)
valDataset   = WMSDataset(valImagePaths,   valMaskPaths,   imageTransforms)
testDataset  = WMSDataset(testImagePaths,  testMaskPaths,  imageTransforms)

# DataLoaders
trainLoader = DataLoader(trainDataset, batch_size=4, shuffle=True)
valLoader   = DataLoader(valDataset,   batch_size=4, shuffle=False)
testLoader  = DataLoader(testDataset,  batch_size=4, shuffle=False)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = WaterMetersUNet(inChannels=3, outChannels=1).to(device)

# Loss, optimizer and scheduler
pos_weight = torch.tensor([1.0], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Tracking
trainLosses, valLosses, testLosses = [], [], []
trainAccs, valAccs, testAccs = [], [], []
numEpochs = 50
bestVal = float('inf')
patienceCtr = 0

# Training loop
for epoch in range(1, numEpochs + 1):
    model.train()
    runningLoss, runningAcc = 0.0, 0.0

    for images, masks in trainLoader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            batch_acc = pixel_accuracy(preds.cpu().numpy(), masks.cpu().numpy())
        runningAcc  += batch_acc
        runningLoss += loss.item()

    avgTrainLoss = runningLoss / len(trainLoader)
    avgTrainAcc  = runningAcc  / len(trainLoader)
    trainLosses.append(avgTrainLoss)
    trainAccs.append(avgTrainAcc)

    # Validation
    model.eval()
    runningLoss, runningValAcc = 0.0, 0.0
    with torch.no_grad():
        for images, masks in valLoader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            runningLoss += criterion(outputs, masks).item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            runningValAcc += pixel_accuracy(preds.cpu().numpy(), masks.cpu().numpy())

    avgValLoss = runningLoss / len(valLoader)
    avgValAcc  = runningValAcc / len(valLoader)
    valLosses.append(avgValLoss)
    valAccs.append(avgValAcc)
    scheduler.step(avgValLoss)

    # Save best result
    if avgValLoss < bestVal:
        bestVal = avgValLoss
        patienceCtr = 0
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', f'best.pth'))
    else:
        patienceCtr += 1
        if patienceCtr >= 5:
            print("Early stopping")
            break

    # Testing
    model.eval()
    runningTestLoss, runningTestAcc = 0.0, 0.0
    with torch.no_grad():
        for images, masks in testLoader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            runningTestLoss += criterion(outputs, masks).item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            runningTestAcc += pixel_accuracy(preds.cpu().numpy(), masks.cpu().numpy())

    avgTestLoss = runningTestLoss / len(testLoader)
    avgTestAcc  = runningTestAcc  / len(testLoader)
    testLosses.append(avgTestLoss)
    testAccs.append(avgTestAcc)

    # Logging
    print(f"Epoch {epoch}/{numEpochs}"
          f" - Train Loss: {avgTrainLoss:.4f}, Train Acc: {avgTrainAcc:.4f}"
          f" - Val Loss: {avgValLoss:.4f}, Val Acc: {avgValAcc:.4f}"
          f" - Test Loss: {avgTestLoss:.4f}, Test Acc: {avgTestAcc:.4f}")

    # Saving checkpoint
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '..', 'models', f'unet_epoch{epoch}.pth'))

# Summary and plots
summary(model, input_size=(3, 512, 512))

plt.figure(figsize=(8,5))
plt.plot(trainLosses, label='Train Loss')
plt.plot(valLosses,   label='Val Loss')
plt.plot(testLosses,  label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.twinx()
plt.plot(trainAccs, '--', label='Train Acc')
plt.plot(valAccs,   '--', label='Val Acc')
plt.plot(testAccs,  '--', label='Test Acc')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Final metrics on test set
print("--- Final evaluation on test set ---")
dice_scores, iou_scores, hausdorff_dists = [], [], []
model.eval()
with torch.no_grad():
    for images, masks in testLoader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float().cpu().numpy()
        masks_np = masks.cpu().numpy()
        for p, m in zip(preds, masks_np):
            dice_scores.append(dice_coeff(p, m))
            iou_scores.append(iou_coeff(p, m))
            p_pts = np.argwhere(p.squeeze()==1)
            m_pts = np.argwhere(m.squeeze()==1)
            hd1 = directed_hausdorff(p_pts, m_pts)[0]
            hd2 = directed_hausdorff(m_pts, p_pts)[0]
            hausdorff_dists.append(max(hd1, hd2))

print(f"Test Dice:      {np.mean(dice_scores):.4f}")
print(f"Test IoU:       {np.mean(iou_scores):.4f}")
print(f"Test Hausdorff: {np.mean(hausdorff_dists):.4f}")


model.eval()
images, masks = next(iter(testLoader))
images = images.to(device)
with torch.no_grad():
    outputs = model(images)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float().cpu()

images = images.cpu().permute(0, 2, 3, 1).numpy()
masks = masks.cpu().squeeze(1).numpy()
preds = preds.squeeze(1).numpy()

for i in range(images.shape[0]):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(images[i])
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(masks[i], cmap='gray')
    plt.title('GT Mask')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(preds[i], cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    plt.show()