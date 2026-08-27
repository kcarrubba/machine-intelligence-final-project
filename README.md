# Smart Fridge Food Classifier — COMP3330/6380 Assessment 1

Deep learning classifier for 40 food item classes using EfficientNet-B0 with
transfer learning and data augmentation.

---

### Full Dataset Access
Due to the large file size (13 GB), the expanded training dataset (10,394 images) is available for download here: 
[https://drive.google.com/file/d/1MnwmRtub7shAAtLzODawh5jGRVberhcu/view?usp=sharing](https://drive.google.com/file/d/1MnwmRtub7shAAtLzODawh5jGRVberhcu/view?usp=sharing)

---

## Requirements

- Python 3.9 or later
- PyTorch 2.0+ (CPU or GPU)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
comp3330-assignment/
├── datasets/
│   └── FoodTest1/            # 143 provided images (flat, NN_ClassName_XXX.JPG)
├── data/
│   ├── raw/                  # additional downloaded images (one subdir per class)
│   └── noise_examples/       # Noise1/2/3 flat folders from Canvas (optional)
├── model/
│   └── best_model.pth        # saved after training
├── train.py                  # training script
├── inference.py              # inference script (standalone)
├── download_data.py          # helper: download extra training images
├── requirements.txt
└── README.md
```

---

## Inference (Evaluating the Submitted Model)

Run inference on any flat dataset folder that follows the `NN_ClassName_XXX.jpg`
naming convention (same as `FoodTest1`):

```bash
python inference.py ./datasets/FoodTest1
```

### Single entrypoint (recommended)

You can run inference (and the metrics/graphs) from one file:

```bash
python3 run.py infer ./datasets/FoodTest1 --metrics --metrics-out metrics
```

### Precision/recall + confusion matrix (false positives / false negatives)

To detect **false positives** (low precision) and **false negatives** (low recall), run:

```bash
python inference.py ./datasets/FoodTest1 --metrics --metrics-out metrics
```

This writes:
- `metrics/precision_recall_f1.csv` (per-class precision/recall/F1/support)
- `metrics/confusion_matrix_counts.png`
- `metrics/confusion_matrix_normalized.png` (row-normalized; easiest to see “what it gets confused with”)

The script automatically loads `model/best_model.pth`. To use a different model:

```bash
python inference.py ./datasets/FoodTest1 --model-path path/to/other_model.pth
```

### Example output

```
Dataset: ./datasets/FoodTest1
Class               Samples  Correct   Accuracy
0_Asparagus               3        2     66.67%
1_Carrotts                4        3     75.00%
...
39_Cheese                 9        8     88.89%
-------------------------------------------------
Mean Class Acc:  72.24%
Overall Acc:     78.57%
```

---

## Training

### 1. Use FoodTest1 only (minimum setup)

```bash
python train.py
```

This trains on the 143 images in `datasets/FoodTest1` with an 80/20 train/val
split and heavy data augmentation.

### 2. With additional structured data (recommended)

Convert the flat `FoodTest1` folder into a standard PyTorch `ImageFolder` style
dataset (one subfolder per class):

```bash
python download_data.py --input datasets/FoodTest1 --output data/raw
```

Then train with the extra data:

```bash
python train.py --extra-data data/raw
```

### 3. With noise examples from Canvas

Download `NoiseExamples.zip` from Canvas, extract to `data/noise_examples/`
(each noise variant is a flat folder with the same naming as FoodTest1), then:

```bash
python train.py --extra-data data/raw --noise-data data/noise_examples
```

### Training arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `datasets/FoodTest1` | Flat folder with provided images |
| `--extra-data` | _(none)_ | Structured folder from `download_data.py` |
| `--noise-data` | _(none)_ | Flat folder with noise example images |
| `--output-dir` | `model` | Where to save `best_model.pth` |
| `--batch-size` | `16` | Training batch size |
| `--phase1-epochs` | `10` | Epochs with frozen backbone |
| `--phase2-epochs` | `40` | Epochs fine-tuning full network |
| `--val-split` | `0.2` | Fraction of data held out for validation |
| `--backbone-lr` | `1e-4` | LR for EfficientNet backbone in phase 2 |
| `--head-lr` | `5e-4` | LR for classifier head in phase 2 |

---

## Model Architecture

- **Base**: EfficientNet-B0 pretrained on ImageNet (via `torchvision`)
- **Head**: `Dropout(0.3) → Linear(1280, 40)`
- **Training**: 2-phase transfer learning
  - Phase 1: backbone frozen, only head trained (AdamW, lr=1e-3, cosine annealing)
  - Phase 2: full network fine-tuned with differential learning rates (backbone lr=1e-4, head lr=5e-4)
- **Loss**: Cross-entropy with label smoothing (0.1)
- **AMP**: Automatic mixed precision on CUDA GPUs

## Data Augmentation

The following transforms are applied during training to improve generalisation
and noise robustness:

- `RandomResizedCrop(224)` — scale 0.65–1.0
- `RandomHorizontalFlip` / `RandomVerticalFlip`
- `RandomRotation(25°)`
- `ColorJitter` (brightness, contrast, saturation, hue)
- `GaussianBlur` — applied with 40% probability (models Noise1/2/3 artefacts)
- `RandomErasing` — simulates occlusion
- ImageNet normalisation

---

## Additional Data Collection

`download_data.py` prepares data into a folder-per-class structure compatible
with `torchvision.datasets.ImageFolder` / `DatasetFolder`:

```bash
# Summary of existing downloaded data
python download_data.py --summary --output data/raw

# Prepare ImageFolder structure from a flat folder like FoodTest1
python download_data.py --input datasets/FoodTest1 --output data/raw
```

---

## Dataset & Licensing

This project uses an expanded dataset (FoodTest1_expanded) combining the provided course dataset with additional images from publicly available sources and manual data collection.

External datasets include:
- Food-101 (Bossard et al., 2014)
- Freiburg Groceries (Jund et al., 2016)
- Fruit Recognition (Mureșan & Oltean, 2018)
- Large-Scale Fish Dataset (Ulucan et al., 2020)
- Hierarchical Grocery Dataset (Klasson et al., 2019)
- Food for Machine Learning Dataset (Fulop & Cristea, 2020)
- Fish-Vista (Mehrab et al., 2025)

All datasets were used in accordance with their respective licenses:
- Food-101: non-commercial scientific use only
- CC BY 4.0: Freiburg Groceries, Fruit Recognition, Fish Dataset
- MIT License: Hierarchical Grocery, FFML, Fish-Vista

This dataset is used strictly for academic, non-commercial purposes. Full details on data sources, preprocessing, and usage are provided in the dataset README.

---

## Hardware Notes

The script auto-selects the best available device:
- **CUDA GPU**: fastest — recommended for training
- **Apple Silicon MPS**: supported (macOS, M1/M2/M3 Macs)
- **CPU**: slowest but always available

Training on CPU with FoodTest1 only (~143 images) typically completes in
under 10 minutes. With additional data and a GPU, expect 30–60 minutes for
50 total epochs.
