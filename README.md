# MLP Model Architecture - Sentiment Analysis

PyTorch-based sentiment analysis project classifying text into **neutral**, **positive**, and **negative** categories.

## Features

- Feedforward neural network with batch normalization and dropout
- Automated CSV preprocessing with feature encoding
- Training pipeline with early stopping and learning rate scheduling
- ONNX conversion for deployment
- WandB experiment tracking

## Installation

**Requirements:** Python 3.12+, `uv` package manager

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Alternative (without uv):**
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib wandb onnx onnxruntime onnxruntime-gpu tqdm torchmetrics
```

## Quick Start

### 0. Dataset
Dataset from huggingface
URL: https://huggingface.co/datasets/mteb/tweet_sentiment_extraction

### 1. Preprocess Data
```bash
cd data/SAAssignment2025
uv run preprocess_csv.py
```
**Output:** `train.npy`, `test.npy`, `val.npy`, `class_names.npy`

### 2. Train Model
```bash
cd ../../SA
uv run SAAssignment2025.py
```
**Output:** `SA-Assingment2025.pth`, `scaler.pkl`, WandB logs

### 3. Convert to ONNX (Optional)
```bash
uv run convert.py
```
**Output:** `SA-Assingment2025.onnx`

### 4. Test ONNX Model (Optional)
```bash
uv run onnxtest.py
```

## Project Structure

```
SA-MLP/
├── data/SAAssignment2025/     # Data preprocessing
│   ├── preprocess_csv.py
│   ├── train.npy, test.npy, val.npy
│   └── class_names.npy
├── SA/                         # Main project
│   ├── SAAssignment2025.py    # Training script
│   ├── model.py               # Neural network architecture
│   ├── preprocess.py          # Data normalization
│   ├── convert.py             # ONNX conversion
│   ├── onnxtest.py            # ONNX testing
│   ├── config.yaml            # Configuration
│   └── SA-Assingment2025.pth  # Trained model
└── README.md
```

## Model Architecture

```
Input (n_features)
  ↓
Linear(256) → BatchNorm → GELU → Dropout(0.2)
  ↓
Linear(128) → BatchNorm → GELU → Dropout(0.2)
  ↓
Output (num_classes)
```

## Configuration

Edit `SA/config.yaml`:

```yaml
class_names:
   - 'neutral'
   - 'positive'
   - 'negative'

batch_size: 128
learning_rate: 0.01
num_epochs: 20
```

**Training Settings:**
- Optimizer: AdamW
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.1)
- Early Stopping: patience=5
- Regularization: Dropout 20%, Gradient clipping (max_norm=1.0)

## Data Format

- **Input:** NumPy arrays with shape `(n_samples, n_features + 1)`
- **Last column:** Integer labels (0, 1, 2, ...)
- **Preprocessing:** StandardScaler normalization (mean=0, std=1)
- **Split:** 80% train, 10% validation, 10% test

## Troubleshooting

**Model file not found:**
- Train model first: `uv run SAAssignment2025.py`

**CUDA errors:**
- Script validates label ranges automatically
- Ensure labels are in `[0, num_classes-1]`

**String conversion errors:**
- Run `preprocess_csv.py` to regenerate encoded data

**ONNX shape mismatch:**
- Regenerate ONNX model: `uv run convert.py`

## Dependencies

- `torch>=2.9.1`, `torchvision>=0.24.1`
- `numpy>=2.3.5`, `pandas>=2.3.3`, `scikit-learn>=1.8.0`
- `onnx>=1.20.0`, `onnxruntime>=1.23.2`
- `wandb>=0.23.1`, `torchmetrics==0.9.3`

## Wandb Evaluation
![Val Loss](MLP-fig/Val_Loss.png)
![Train Loss](MLP-fig/Train_Loss.png)
![Learning Rate](MLP-fig/Learning_Rate.png)
![Accuracy](MLP-fig/Accuracy.png)


