from torch.backends import mps
from torch import cuda
from pathlib import Path

device = "cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu"

root_dir = Path(__file__).parent.parent

data_dir = root_dir / "data"
data_dir.mkdir(parents=True,exist_ok=True)

models_dir = root_dir / "models"
models_dir.mkdir(parents=True,exist_ok=True)

	