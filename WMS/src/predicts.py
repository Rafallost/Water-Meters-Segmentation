import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import WaterMetersUNet
from transforms import imageTransforms
import cv2

# Ścieżka do wag modelu
model_path = "../models/unet_trained_binary.pth"

# Folder z obrazami do przewidzenia
image_folder = "../data/test/images"

# Folder do zapisu masek
save_folder = "../predictions"
os.makedirs(save_folder, exist_ok=True)

# Ustawienie urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wczytaj model
model = WaterMetersUNet(inChannels=3, outChannels=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Przekształcenia jak w treningu
transform = imageTransforms

# Pętla po obrazach testowych
for filename in os.listdir(image_folder):
    if not filename.endswith(".jpg"):
        continue

    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(transformed)
        prediction = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()
        prediction_img = (prediction * 255).astype(np.uint8)

    # Zapis maski
    save_path = os.path.join(save_folder, f"mask_{filename}")
    Image.fromarray(prediction_img).save(save_path)

    # (opcjonalnie) pokaż obraz + maskę
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Obraz")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(prediction_img, cmap="gray")
    plt.title("Maska przewidziana")
    plt.axis("off")
    plt.show()
