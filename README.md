# ML Foundations

A FashionMNIST image classifier built with PyTorch. Trains a fully-connected neural network to classify 28×28 grayscale images into 10 clothing categories.

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

## Project Structure

```
├── train.py           # Training and evaluation loops
├── neuralnetwork.py   # Model architecture
├── data.py            # Dataset loading
└── utils/
    └── setup.py       # Shared config (device, directories)
```
