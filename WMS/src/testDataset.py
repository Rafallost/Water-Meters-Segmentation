import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import WMSDataset
from transforms import imageTransforms, maskTransforms

# Ścieżka do bieżącego pliku
script_dir = os.path.dirname(os.path.abspath(__file__))

image_dir = os.path.join(script_dir, '..', 'data', 'images')
mask_dir  = os.path.join(script_dir, '..', 'data', 'masks')

# Pobieramy listę plików
imagePaths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
maskPaths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]

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

plt.show()

