import torchvision.transforms as TRANS

imageTransforms = TRANS.Compose([
    TRANS.ToPILImage(),
    TRANS.Resize((512, 512)), #512/4        768/2
    TRANS.ToTensor(),
    TRANS.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Masks contain classes, not pixel values - without normalization
maskTransforms = TRANS.Compose([
    TRANS.ToPILImage(),
    TRANS.Resize((512, 512)), #512/4        768/2
    TRANS.ToTensor()
])
