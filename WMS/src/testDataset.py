import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import WMSDataset
from transforms import imageTransforms, maskTransforms

DATASET_PATH  = os.path.dirname(os.path.abspath(__file__))
IMAGE_DATASET_PATH  = os.path.join(DATASET_PATH , '..', 'data', 'images')
MASK_DATASET_PATH   = os.path.join(DATASET_PATH , '..', 'data', 'masks')

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pobieramy listę plików
imagePaths = [os.path.join(IMAGE_DATASET_PATH , f) for f in os.listdir(IMAGE_DATASET_PATH )]
maskPaths = [os.path.join(MASK_DATASET_PATH , f) for f in os.listdir(MASK_DATASET_PATH )]

dataset = WMSDataset(imagePaths, maskPaths, imageTransforms, maskTransforms)

print(f"Dataset length: {len(dataset)}")

dataLoader = DataLoader(dataset, batch_size=5, shuffle=True)

images, masks = next(iter(dataLoader))

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


print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"cuda version: {torch.version.cuda}")

#plt.show()

