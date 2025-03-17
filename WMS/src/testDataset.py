from dataset import WMSDataset

dataset = WMSDataset()
image, mask = dataset[10]

print("Typ image:", type(image))
print("Typ mask:", type(mask))

image.show()
mask.show()