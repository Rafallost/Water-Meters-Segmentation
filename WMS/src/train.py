import os
import random

import torch
from scipy.ndimage import value_indices
from torch import nn, optim
from torch.utils.data import DataLoader

from WMS.src.model import UNet
from WMS.src.testDataset import IMAGE_DATASET_PATH, MASK_DATASET_PATH, imagePaths, image
from dataset import WMSDataset
from transforms import imageTransforms, maskTransforms

IMAGE_DATASET_PATH = r"../data/images"
MASK_DATASET_PATH = r"../data/masks"

# Read data
imagePaths = sorted([os.path.join(IMAGE_DATASET_PATH, f) for f in os.listdir(IMAGE_DATASET_PATH)])
maskPaths = sorted([os.path.join(MASK_DATASET_PATH, f) for f in os.listdir(MASK_DATASET_PATH)])

indices = list(range(len(imagePaths)))
random.shuffle(indices)

# Split size: 80/10/10
trainSize = int(0.8 * len(indices))
valSize = int(0.1 * len(indices))
testSize = int(0.1 * len(indices))

# Split all indices
trainIndices = indices[:trainSize]
valueIndices = indices[trainSize:trainSize + valSize]
testIndices = indices[trainSize + valSize:]

# Paths for sub datasets
trainImagePaths = [imagePaths[i] for i in trainIndices]
trainMaskPaths = [maskPaths[i] for i in trainIndices]
valImagePaths = [imagePaths[i] for i in valueIndices]
valMaskPaths = [maskPaths[i] for i in valueIndices]
testImagePaths = [imagePaths[i] for i in testIndices]
testMaskPaths = [maskPaths[i] for i in testIndices]

# Creating objects
trainDataset = WMSDataset(trainImagePaths, trainMaskPaths, imageTransforms, maskTransforms)
valDataset = WMSDataset(valImagePaths, valMaskPaths, imageTransforms, maskTransforms)
testDataset = WMSDataset(testImagePaths, testMaskPaths, imageTransforms, maskTransforms)

# Creating dataloaders for each object
trainLoader = DataLoader(trainDataset, batch_size=8, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=8, shuffle=False)
testLoader = DataLoader(testDataset, batch_size=8, shuffle=False)

print(f"Train samples: {len(trainDataset)}")
print(f"Validate samples:   {len(valDataset)}")
print(f"Test samples:  {len(testDataset)}")

# Turn on cuda if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = UNet(nbClasses=1, outSize=(256,256))
model.to(device)

# Cost function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

numEpochs = 25

for epoch in range(numEpochs):
    model.train()
    runningLoss = 0.0

    for images, masks in trainLoader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()

    avgTrainLoss = runningLoss / len(trainLoader)

    model.eval()
    runningLoss = 0.0
    with torch.no_grad():
        for images, masks in valLoader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            runningLoss += loss.item()

    avgValLoss = runningLoss / len(valLoader)
    print(f"Epoch {epoch + 1}/{numEpochs} - Train Loss: {avgTrainLoss:.4f} - Val Loss: {avgValLoss:.4f}")

    torch.save(model.state_dict(), "../models/unet_trained.pth")