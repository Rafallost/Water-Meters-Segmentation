import torchvision.transforms as TRANS

imageTransforms = TRANS.Compose([
    TRANS.ToPILImage(),
    TRANS.Resize((256, 256)),
    #TRANS.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # change colors
    TRANS.ToTensor()
    #TRANS.Normalize(mean=[0.5], std=[0.5])  # Normalization to [-1,1]
])

# Masks contain classes, not pixel values - without normalization
maskTransforms = TRANS.Compose([
    TRANS.ToPILImage(),
    TRANS.Resize((256, 256)),
    #TRANS.RandomHorizontalFlip(p=0.5),
    #TRANS.RandomRotation(15),
    TRANS.ToTensor()
])
