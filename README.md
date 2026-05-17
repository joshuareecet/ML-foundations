# ML Foundations

A FashionMNIST image classifier built with PyTorch. Trains and evaluates two architectures — a fully-connected network and a CNN — to classify 28×28 grayscale images into 10 clothing categories.

## Setup

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train.py
```

The dataset downloads automatically on first run. The trained model is saved to `models/`.

To switch between architectures, change the initialiser call in `train.py`:

```python
# Fully-connected network
model, loss_fn, optimizer, scheduler = neural_net_init()

# CNN (default)
model, loss_fn, optimizer, scheduler = CNN_init()
```

## Models

| Name | Architecture | Optimizer | Saved as |
|------|-------------|-----------|----------|
| `NeuralNetwork` | 3-layer MLP (512-512-10) | SGD |
| `CNN` | 3× Conv2d + BN + MaxPool → FC(256) → 10 | AdamW |

The CNN uses batch normalization and dropout (conv: 0.1, FC: 0.2) for regularization.

## Data Pipeline

`data.py` provides:

- **`CustomImageDataset`** — loads images from a CSV annotations file and image directory
- **`TransformedSubset`** — wraps a `torch.utils.data.Subset` with separate transforms per split
- **`train_val_split`** — stratified 80/20 train/validation split using scikit-learn

Training augmentations applied to the train split: random rotation (±6°), random horizontal flip (15%), random crop with padding 4.

## Training

Default hyperparameters in `train.py`:

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| LR schedule | Linear decay to 0.5× over 30 epochs |

## Project Structure

```
├── train.py           # Training and evaluation loops, model initialisers
├── neuralnetwork.py   # NeuralNetwork (MLP) and CNN architectures
├── data.py            # Dataset classes, splitter, and FMNIST loader
├── models/            # Saved model state dicts
└── utils/
    └── setup.py       # Device selection (CUDA / MPS / CPU) and shared paths
```
