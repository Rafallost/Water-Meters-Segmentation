import torchvision.transforms as TRANS

imageTransforms = TRANS.Compose([
    TRANS.ToPILImage(),
    TRANS.Resize((512, 512)), #512/4        768/2
    TRANS.ToTensor()
])

# Masks contain classes, not pixel values - without normalization
maskTransforms = TRANS.Compose([
    TRANS.ToPILImage(),
    TRANS.Resize((512, 512)), #512/4        768/2
    TRANS.ToTensor()
])
