import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

""" 
os.listdir - return a list containing the names of the files in the directory
os.path.join - a full path by concatenating various components while automatically inserting the appropriate path separator
"""

class WMSDataset(Dataset):
    def __init__(self, transformImage=None, transformMask=None):
        self.image_dir = r"D:\Github\Water-Meters-Segmentation\WMS\data\images"
        self.mask_dir = r"D:\Github\Water-Meters-Segmentation\WMS\data\masks"
        self.imageNames = os.listdir(self.image_dir)
        self.maskNames = os.listdir(self.mask_dir)
        self.transformImage = transformImage
        self.transformMask = transformMask

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, i):
        image_path = os.path.join(self.image_dir, self.imageNames[i])
        mask_path = os.path.join(self.mask_dir, self.maskNames[i])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transformImage:
            image = self.transformImage(image)
        if self.transformMask:
            mask = self.transformMask(mask)

        return image, mask


