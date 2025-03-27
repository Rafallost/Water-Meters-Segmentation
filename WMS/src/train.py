import os
import random

from scipy.ndimage import value_indices
from torch.utils.data import DataLoader

from WMS.src.testDataset import IMAGE_DATASET_PATH, MASK_DATASET_PATH, imagePaths
from dataset import WMSDataset
from transforms import imageTransforms, maskTransforms

IMAGE_DATASET_PATH = r"../data/images"
MASK_DATASET_PATH = r"../data/masks"

imagePaths = [os.path.join(IMAGE_DATASET_PATH, i) for i in os.listdir((IMAGE_DATASET_PATH))]
maskPaths = [os.path.join(MASK_DATASET_PATH, i) for i in os.listdir((MASK_DATASET_PATH))]

indices = list(range(len(imagePaths)))
random.shuffle(indices)

trainSize = int(0.8 * len(indices))
valSize = int(0.1 * len(indices))
testSize = int(0.1 * len(indices))

trainIndices = indices[:trainSize]
valueIndices = indices[trainSize:trainSize + valSize]
testIndices = indices[trainSize + valSize]