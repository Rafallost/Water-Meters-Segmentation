import os
import shutil
import random
from collections import defaultdict

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import WMSDataset
from transforms import imageTransforms, maskTransforms

############### DATA LOAD ###############
random.seed(42)

# Source data directories
datasetPath = os.path.dirname(os.path.abspath(__file__))
sourceImageDir = os.path.join(datasetPath, '..', 'data', 'images')
sourceMaskDir  = os.path.join(datasetPath, '..', 'data', 'masks')

# Get images and masks names
imageFiles = sorted([f for f in os.listdir(sourceImageDir) if f.endswith('.jpg')])
maskFiles = sorted([f for f in os.listdir(sourceMaskDir) if f.endswith('.jpg')])
assert len(imageFiles) == len(maskFiles), "Amount of images and masks have to be the same"

# 80% train, 10% val, 10% test
trainImgs, tempImgs = train_test_split(imageFiles, test_size=0.2, random_state=42)
valImgs, testImgs = train_test_split(tempImgs, test_size=0.5, random_state=42)

splits = {'train': trainImgs, 'val': valImgs, 'test': testImgs}

# Folders creation
baseDataDir = os.path.join(datasetPath, '..', 'data')
for split, files in splits.items():
    for subfolder in ['images', 'masks']:
        os.makedirs(os.path.join(baseDataDir, split, subfolder), exist_ok=True)
    for fname in files:
        shutil.copy(os.path.join(sourceImageDir, fname), os.path.join(baseDataDir, split, 'images', fname))
        shutil.copy(os.path.join(sourceMaskDir, fname), os.path.join(baseDataDir, split, 'masks', fname))

os.makedirs("../models", exist_ok=True)

# Load data from folder 'train'
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

############### DATA VERIFICATION ###############
def count_pixel_balance(mask_paths, dataset_name):
    counts = defaultdict(int)
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, 0)  # Wczytujemy maskę jako grayscale
        if mask is None:
            print(f"Warning: Could not load {mask_path}")
            continue
        unique, cnts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, cnts):
            counts[cls] += count
    print(f"\nPixel distribution for the set {dataset_name}:")
    for cls, count in sorted(counts.items()):
        print(f"Class {cls}: {count} pixels")

    # Wykres słupkowy dla wizualizacji
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]
    plt.figure(figsize=(8, 6))
    plt.bar([str(c) for c in classes], values)
    plt.xlabel("Class")
    plt.ylabel("number of pixels")
    plt.title(f"Pixel distribution for the set: {dataset_name}")
    plt.show()


# Liczymy balance dla poszczególnych zbiorów
count_pixel_balance(trainMaskPaths, "Train")
count_pixel_balance(valMaskPaths, "Validation")
count_pixel_balance(testMaskPaths, "Test")

############### DEVICE CONFIGURATION ###############
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False


############### PLOTS ###############
import matplotlib.pyplot as plt

sizes = [len(trainImgs), len(valImgs), len(testImgs)]
labels = ['Train', 'Validation', 'Test']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Podział zbioru danych')
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.show()

fig, axs = plt.subplots(5, 2, figsize=(10, 20))
for i in range(5):
    image = images[i].permute(1, 2, 0).numpy()
    mask = masks[i].squeeze().numpy()
    axs[i, 0].imshow(image)
    axs[i, 0].set_title("Obraz")
    axs[i, 1].imshow(mask, cmap='gray')
    axs[i, 1].set_title("Maska")
    axs[i, 0].axis("off")
    axs[i, 1].axis("off")


print(f"\nPyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"cuda version: {torch.version.cuda}")

plt.show()