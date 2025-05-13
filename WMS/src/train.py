import subprocess
import os
import sys
import torch
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from src.model import WaterMetersUNet
from dataset import WMSDataset

from transforms import imageTransforms, maskTransforms

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

# Prepare data
prepare_script = os.path.join(os.path.dirname(__file__), 'prepareDataset.py')
subprocess.run([sys.executable, prepare_script], check=True)

# Load data
baseDataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
trainImagePaths = [os.path.join(baseDataDir, 'train', 'images', f)
                   for f in os.listdir(os.path.join(baseDataDir, 'train', 'images')) if f.endswith('.jpg')]
trainMaskPaths  = [os.path.join(baseDataDir, 'train', 'masks', f)
                   for f in os.listdir(os.path.join(baseDataDir, 'train', 'masks')) if f.endswith('.jpg')]

testImagePaths = [os.path.join(baseDataDir, 'test', 'images', f)
                   for f in os.listdir(os.path.join(baseDataDir, 'test', 'images')) if f.endswith('.jpg')]
testMaskPaths  = [os.path.join(baseDataDir, 'test', 'masks', f)
                   for f in os.listdir(os.path.join(baseDataDir, 'test', 'masks')) if f.endswith('.jpg')]

valImagePaths = [os.path.join(baseDataDir, 'val', 'images', f)
                   for f in os.listdir(os.path.join(baseDataDir, 'val', 'images')) if f.endswith('.jpg')]
valMaskPaths  = [os.path.join(baseDataDir, 'val', 'masks', f)
                   for f in os.listdir(os.path.join(baseDataDir, 'val', 'masks')) if f.endswith('.jpg')]

trainDataset = WMSDataset(trainImagePaths, trainMaskPaths, imageTransforms, maskTransforms)
testDataset = WMSDataset(testImagePaths, testMaskPaths,  imageTransforms, maskTransforms)
valDataset  = WMSDataset(valImagePaths,   valMaskPaths,   imageTransforms, maskTransforms)

print(f"trainDataset length(train part): {len(trainDataset)}")
print(f"testDataset length(train part): {len(testDataset)}")
print(f"valDataset length(train part): {len(valDataset)}")

# Get one batch
dataLoader = DataLoader(trainDataset, batch_size=5, shuffle=True)
images, masks = next(iter(dataLoader))

# Creating dataloaders for each object
trainLoader = DataLoader(trainDataset, batch_size=4, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=4, shuffle=False)
testLoader = DataLoader(testDataset, batch_size=4, shuffle=False)

print(f"Train samples: {len(trainDataset)}")
print(f"Validate samples:   {len(valDataset)}")
print(f"Test samples:  {len(testDataset)}")

# Turn on cuda if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model (in RGB out binary)
model = WaterMetersUNet(inChannels=3, outChannels=1)
model.to(device)

# Binary Cross Entropy with Logits - loss function for binary classification
# Takes model output (logits), applies sigmoid internally, and computes the loss
criterion = nn.BCEWithLogitsLoss()

# Adam optimizer with a learning rate of 5e-5 (slow, stable training)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

testLosses = []
trainLosses = []
valLosses   = []

numEpochs = 10
for epoch in range(numEpochs):
    model.train()  # Set model to training mode (activates Dropout, updates BatchNorm)
    runningLoss = 0.0

    # Iterate over the entire training dataset in batches
    for images, masks in trainLoader:
        images = images.to(device)
        masks = masks.to(device)
        # Clear gradients from the previous step
        optimizer.zero_grad()
        # Forward pass through the network
        outputs = model(images)
        # Calculate loss between model predictions and ground truth masks
        loss = criterion(outputs, masks)
        # Backpropagation: compute gradients
        loss.backward()
        # Update model weights based on gradients
        optimizer.step()
        runningLoss += loss.item()

    # Compute average training loss for this epoch
    avgTrainLoss = runningLoss / len(trainLoader)
    trainLosses.append(avgTrainLoss)

    # Set model to evaluation mode (disables Dropout, uses running stats in BatchNorm)
    model.eval()
    runningLoss = 0.0
    # Inference loop - same as training, but without gradient tracking
    with torch.no_grad():
        for images, masks in valLoader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            runningLoss += loss.item()

    # Compute average validation loss for this epoch
    avgValLoss = runningLoss / len(valLoader)
    valLosses.append(avgValLoss)
    print(f"Epoch {epoch + 1}/{numEpochs} - Train Loss: {avgTrainLoss:.4f} - Val Loss: {avgValLoss:.4f}")

    runningTestLoss = 0.0
    model.eval()
    with torch.no_grad():
        for images, masks in testLoader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            runningTestLoss += loss.item()
    avgTestLoss = runningTestLoss / len(testLoader)
    testLosses.append(avgTestLoss)
    print(f"Epoch {epoch + 1}/{numEpochs} - Test Loss: {avgTestLoss:.4f}")

    # Save model weights to file
    torch.save(model.state_dict(), "../models/unet_trained_binary.pth")

# One RGB image 512x512
summary(model, input_size=(3, 512, 512))

plt.figure(figsize=(7,4))
plt.plot(trainLosses, label='Train loss')
plt.plot(valLosses,   label='Val loss')
plt.plot(testLosses,  label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('BCE/Dice loss')
plt.title('Learning curve')
plt.legend()
plt.grid()
plt.show()

# --- oblicz metryki na ZBIORZE TESTOWYM ---
dice_scores = []
iou_scores  = []
hausdorff_dists = []

model.eval()
with torch.no_grad():
    for images, masks in testLoader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float().cpu().numpy()
        masks_np = masks.cpu().numpy()

        for p, m in zip(preds, masks_np):
            # Dice, IoU
            dice_scores.append(dice_coeff(p, m))
            iou_scores.append(iou_coeff(p, m))
            # Hausdorff (dwukierunkowo)
            p_pts = np.argwhere(p.squeeze()==1)
            m_pts = np.argwhere(m.squeeze()==1)
            hd1 = directed_hausdorff(p_pts, m_pts)[0]
            hd2 = directed_hausdorff(m_pts, p_pts)[0]
            hausdorff_dists.append(max(hd1, hd2))

print(f"Test Dice:    {np.mean(dice_scores):.4f}")
print(f"Test IoU:     {np.mean(iou_scores):.4f}")
print(f"Test Hausdorff: {np.mean(hausdorff_dists):.4f}")


