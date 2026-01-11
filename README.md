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
├── .gitignore                    # Git ignore configuration
├── README.md                     # This file
├── Results/
│   ├── custom_images/            # Self-taken photos of water meters 
│   ├── custom_predictions/       # Predicted masks for self-taken photos
│   ├── Report_EN.md              # Comprehensive project report
│   ├── Report_PL.pdf             # Original Polish report (obsolete)
│   ├── Example_Prediction_*.png  # Example predictions (4 files)
│   ├── Pixel_Distribution_*.png  # Dataset statistics (3 files)
│   ├── Distribution_Set_Plot.png # Dataset split visualization
│   ├── plot_*.png                # Training curves (2 files)
│   └── Terminal.log              # Training output log
└── WMS/
    ├── data/
    │   ├── training/
    │   │   ├── images/           # [REQUIRED] Source images
    │   │   ├── masks/            # [REQUIRED] Source masks
    │   │   └── temp/             # [AUTO-GENERATED] Train/val/test splits
    │   │       ├── train/        #  80% of source
    │   │       ├── val/          #  10% of source
    │   │       └── test/         #  10% of source
    │   └── predictions/
    │       ├── photos_to_predict/  # [USER INPUT] Images to predict
    │       └── predicted_masks/    # [AUTO-GENERATED] Output masks
    ├── models/
    │   ├── best.pth              # [AUTO-GENERATED] Best checkpoint
    │   └── unet_epoch*.pth       # [AUTO-GENERATED] Epoch checkpoints
    └── src/
        ├── dataset.py            # PyTorch Dataset class
        ├── model.py              # U-Net architecture
        ├── transforms.py         # Image preprocessing
        ├── prepareDataset.py     # Data splitting (80/10/10)
        ├── train.py              # Training loop + metrics
        └── predicts.py           # Inference (no temp dirs needed)
```

**Key Notes:**
- `[REQUIRED]` directories must exist with data before training
- `[AUTO-GENERATED]` directories are created automatically by scripts
- `[USER INPUT]` directories are where you place new images for predictions
- The `temp/` directory is ignored by git (temporary training splits)
- `predicts.py` works independently with just `best.pth` and input images

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

### Training Workflow

**Prerequisites:**
- Place your dataset in `WMS/data/training/`:
  - Images: `WMS/data/training/images/*.jpg`
  - Masks: `WMS/data/training/masks/*.jpg`

**1. Prepare Dataset (optional)**
```bash
python WMS/src/prepareDataset.py
```
- Splits data into train/val/test sets (80%/10%/10%)
- Creates temporary directories in `WMS/data/training/temp/`
- **Not required** - `train.py` runs this automatically

**2. Train Model**
```bash
python WMS/src/train.py
```
- Automatically runs `prepareDataset.py` first
- Trains U-Net for 50 epochs with early stopping
- Saves checkpoints to `WMS/models/`:
  - `best.pth` - Best model based on validation loss
  - `unet_epoch{N}.pth` - Checkpoint for each epoch
- Evaluates on train/val/test sets each epoch
- Displays training plots and sample predictions

### Inference Workflow

**Prerequisites:**
- A trained model: `WMS/models/best.pth`
- Input images: Place JPG files in `WMS/data/predictions/photos_to_predict/`

**Run Predictions**
```bash
python WMS/src/predicts.py
```
- Loads the best trained model
- Processes all `.jpg` files in `photos_to_predict/`
- Displays side-by-side comparisons (original | predicted mask)
- Saves predicted masks to `WMS/data/predictions/predicted_masks/`
- **No training required** - works with pre-trained models from GitHub

## Authors

- **Wojciech Szewczyk**
- **Rafal Zablotni**

## Documentation

- **[Report_EN.md](Results/Report_EN.md)** - Comprehensive project report in English with current results
- **[Report_PL.pdf](Results/Report_PL.pdf)** - Original academic report in Polish (contains obsolete results)
