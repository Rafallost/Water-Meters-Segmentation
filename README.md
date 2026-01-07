# Water Meters Segmentation

A deep learning project for segmenting water meter displays using U-Net architecture.

**Course**: Fundamentals of Artificial Intelligence

## Example

<img src="Results/Example_Prediction_4.png" width="1200" alt="Example water meter image">

## Dataset

<a href="https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset?resource=download-directory">Water Meters Dataset</a> (KUCEV ROMAN)

| Property | Value |
|----------|-------|
| Images | 1244 |
| Masks | 1244 |
| Format | JPG |
| Image type | RGB |
| Mask type | Grayscale |
| Scaled size | 512x512 |

**Data Split:**
- Train: 995 images (80%)
- Validation: 124 images (10%)
- Test: 125 images (10%)

## Model Architecture

U-Net encoder-decoder architecture for binary segmentation.

| Component | Description |
|-----------|-------------|
| Encoder | 4 levels with Conv2d + BatchNorm2d + ReLU + MaxPool2d |
| Bottleneck | Deepest feature representation |
| Decoder | 4 levels with upsampling + skip connections |
| Output | 1 channel binary mask |

| Parameter | Value |
|-----------|-------|
| Input channels | 3 (RGB) |
| Output channels | 1 (binary mask) |
| Input size | 512x512 |
| Total parameters | 243,425 |
| Trainable parameters | 243,425 |

## Results

| Metric | Value |
|--------|-------|
| Test Dice Coefficient | 0.7762 |
| Test IoU (Intersection over Union) | 0.6842 |
| Test Hausdorff Distance | 129.2380 |
| Test Pixel Accuracy | 99.11% |
| Final Train Loss | 0.0204 |
| Final Val Loss | 0.0401 |
| Final Test Loss | 0.0309 |

Training was conducted over 50 epochs. Both train and validation losses decreased consistently, indicating good generalization without overfitting.

### Training Progress (Selected Epochs)

| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | Test Loss | Test Dice |
|-------|------------|------------|----------|----------|-----------|-----------|
| 1     | 0.5028     | 0.2056     | 0.4609   | 0.2231   | 0.4599    | 0.2193    |
| 10    | 0.2362     | 0.5260     | 0.2314   | 0.5451   | 0.2305    | 0.5462    |
| 20    | 0.1058     | 0.6538     | 0.1113   | 0.6373   | 0.1074    | 0.6484    |
| 30    | 0.0507     | 0.7314     | 0.0574   | 0.6723   | 0.0512    | 0.7012    |
| 40    | 0.0288     | 0.7941     | 0.0429   | 0.6368   | 0.0359    | 0.6617    |
| 50    | 0.0204     | 0.8289     | 0.0401   | 0.6544   | 0.0309    | 0.6783    |

## Project Structure

```
Water-Meters-Segmentation/
├── README.md
├── Results/
│   ├── Report_EN.md            # Current project report in English
│   ├── Report_PL.pdf           # Original academic report in Polish (obsolete results)
│   ├── Example_Prediction_*.png  # Example predictions (4 files)
│   ├── Pixel_Distribution_*.png  # Dataset statistics visualizations
│   ├── Distribution_Set_Plot.png # Dataset split visualization
│   ├── plot_*.png              # Additional training plots
│   └── Terminal.log            # Training output log
└── WMS/
    ├── data/
    │   ├── images/             # Original images
    │   ├── masks/              # Segmentation masks
    │   └── collage/            # Sample visualizations
    ├── models/
    │   ├── best.pth            # Best model checkpoint
    │   └── unet_epoch*.pth     # Epoch checkpoints (50 files)
    └── src/
        ├── dataset.py          # WMSDataset class
        ├── model.py            # U-Net model implementation
        ├── transforms.py       # Image preprocessing transforms
        ├── prepareDataset.py   # Data preparation and splitting
        ├── train.py            # Training script
        └── predicts.py         # Inference script
```

## Requirements

- Python 3.x
- PyTorch 2.6.0+cu118
- Torchvision 0.21.0+cpu
- CUDA 11.8
- OpenCV (cv2)
- NumPy
- SciPy
- scikit-learn
- matplotlib
- torchsummary

## Usage

### 1. Prepare Dataset (optional)
```bash
python WMS/src/prepareDataset.py
```
This script splits the data into train/val/test sets.
It does not need to be run separately, as WMS/src/train.py runs it automatically before training.

### 2. Train Model
```bash
python WMS/src/train.py
```
Trains the U-Net model and saves checkpoints to `models/`.

### 3. Run Inference
```bash
python WMS\src\predicts.py
```
Runs predictions on test images or custom images.

## Authors

- **Wojciech Szewczyk**
- **Rafal Zablotni**

## Documentation

- **[Report_EN.md](Results/Report_EN.md)** - Comprehensive project report in English with current results
- **[Report_PL.pdf](Results/Report_PL.pdf)** - Original academic report in Polish (contains obsolete results)
