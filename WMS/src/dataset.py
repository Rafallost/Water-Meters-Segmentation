
from torch.utils.data import Dataset # All PyTorch datasets must inherit from this base dataset class
import cv2

""" 

"""

class WMSDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, i):
        image_path = self.imagePaths[i] #Grab the image path form the current index
        image = cv2.imread(image_path) # Load image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Swap channels from BGR to RGB

        mask_path = self.maskPaths[i]
        mask = cv2.imread(mask_path, 0) # 0 - greyscale

        # Normalization
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask


