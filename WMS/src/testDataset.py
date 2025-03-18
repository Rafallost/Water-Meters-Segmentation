from random import random

from dataset import WMSDataset
import os
import cv2

image_dir = r"D:\Github\Water-Meters-Segmentation\WMS\data\images"
mask_dir = r"D:\Github\Water-Meters-Segmentation\WMS\data\masks"

imagePaths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
maskPaths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]

transforms = None  # torchvision.transforms, to tensor

dataset = WMSDataset(imagePaths, maskPaths, transforms)

image, mask = dataset[0]

cv2.imshow("Image", image)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()