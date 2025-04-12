import torch
import torch.nn as nn # Conv2d, BatchNorm2d etc.
import torch.nn.functional as F # interpolate

class WaterMetersUNet(nn.Module):
    def __init__(self, inChannels, baseFilters=16, outChannels=1):
        super(WaterMetersUNet, self).__init__()

        # ENCODER
        self.enc1 = nn.Sequential(
            nn.Conv2d(inChannels, baseFilters, kernel_size=3, padding=1),
            nn.BatchNorm2d(baseFilters),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, stride=2) # Resize h, w by half (e.g. 512->256)


        self.enc2 = nn.Sequential(
            nn.Conv2d(baseFilters, baseFilters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(baseFilters * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)


        self.enc3 = nn.Sequential(
            nn.Conv2d(baseFilters * 2, baseFilters * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(baseFilters * 4),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # Deepest layer of UNet, without pooling after that
        self.bottleneck = nn.Sequential(
            nn.Conv2d(baseFilters * 4, baseFilters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(baseFilters * 8),
            nn.ReLU(inplace=True)
        )

