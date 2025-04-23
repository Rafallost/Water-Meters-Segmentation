import subprocess
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from WMS.src.model import WaterMetersUNet

from dataset import WMSDataset
from transforms import imageTransforms, maskTransforms

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
testDataset = WMSDataset(testImagePaths, trainMaskPaths, imageTransforms, maskTransforms)
valDataset = WMSDataset(valImagePaths, trainMaskPaths, imageTransforms, maskTransforms)

print(f"trainDataset length(train part): {len(trainDataset)}")
print(f"testDataset length(train part): {len(testDataset)}")
print(f"valDataset length(train part): {len(valDataset)}")

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

# Cost function, Binary Cross Entropy z logitami, czyli bez sigmoid na końcu modelu
criterion = nn.BCEWithLogitsLoss()

# Optimizer to set learning speed. 5e-5 means slow training
optimizer = optim.Adam(model.parameters(), lr=5e-5)

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