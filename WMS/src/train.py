import subprocess
import os
import sys
import torch
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary

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

trainLosses = []
valLosses   = []

numEpochs = 25
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

    # Save model weights to file
    torch.save(model.state_dict(), "../models/unet_trained_binary.pth")

# One RGB image 512x512
summary(model, input_size=(3, 512, 512))

plt.figure(figsize=(7,4))
plt.plot(trainLosses, label='Train loss')
plt.plot(valLosses,   label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('BCE/Dice loss')
plt.title('Learning curve')
plt.legend()
plt.grid()
plt.show()

